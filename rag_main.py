import logging
import os
import sqlite3
import hashlib
import threading
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

class RAGsys:

    def __init__(self, default_dir="./data"):
        self.lock = threading.RLock()
        self.base_dir = default_dir
        self.docs_dir = os.path.join(default_dir,"docs")
        self.db_path = os.path.join(default_dir,"db.sqlite3")
        self.faiss_path = os.path.join(default_dir,"index","faiss.index")
        self.topics = ["engineering","literature","science"]
        self.conn = None
        self.index = None
        self.encoder = None
        self.observer = None
        self.model_name = "all-MiniLM-L6-v2"
        self.dimension = 384
        self.target_tokens = 200
        self.max_tokens = 256
        self.overlap_tokens = 40
        self.next_id = 0

        logger.info("Starting RAG system")

        self._setup()
        self._load_or_create_index()
        self._index_all()
        self._start_watcher()

        logger.info("System ready")

    def _setup(self):
        os.makedirs(self.docs_dir,exist_ok=True)
        os.makedirs(os.path.dirname(self.faiss_path),exist_ok=True)
        os.makedirs(self.base_dir,exist_ok=True)

        for topic in self.topics:
            os.makedirs(os.path.join(self.docs_dir,topic),exist_ok=True)

        logger.info(f"Loading embedding model: "f"{self.model_name}")

        self.encoder = SentenceTransformer(self.model_name)

        actual_dimension = (self.encoder.get_sentence_embedding_dimension())

        if actual_dimension != self.dimension:
            self.dimension = actual_dimension
            logger.warning(f"Embedding dimension adjusted to "f"{self.dimension}")

        self.conn = sqlite3.connect(self.db_path,check_same_thread=False)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                faiss_id INTEGER UNIQUE,
                file_path TEXT,
                file_name TEXT,
                topic TEXT,
                chunk_index INTEGER,
                start_pos INTEGER,
                end_pos INTEGER,
                chunk_text TEXT,
                file_hash TEXT,
                created_at TIMESTAMP
                    DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_topic
            ON chunks(topic)
        """)

        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_faiss_id
            ON chunks(faiss_id)
        """)

        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_file_path
            ON chunks(file_path)
        """)

        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_file_hash
            ON chunks(file_hash)
        """)

        self.conn.commit()

        logger.info("Database setup complete")

    def _create_empty_index(self):

        logger.info("Creating new FAISS FlatIP index")

        base_index = faiss.IndexFlatIP(self.dimension)

        # base_index.hnsw.efConstruction = 200
        # base_index.hnsw.efSearch = 64

        self.index = faiss.IndexIDMap2(base_index)
        self.next_id = 0

    def _load_or_create_index(self):

        if os.path.exists(self.faiss_path):
            try:
                logger.info("Loading existing FAISS index")

                self.index = faiss.read_index(self.faiss_path)

                self.next_id = (self._get_next_id())

                logger.info(f"Loaded FAISS index: "f"{self.index.ntotal} vectors")
            except Exception as e:
                logger.error(f"Failed to load FAISS index: {e}")
                logger.info("Creating a new FAISS index")
                self._create_empty_index()
        else:
            logger.info("No FAISS index found")
            self._create_empty_index()

    def _get_next_id(self):
        row = self.conn.execute(
            """
            SELECT MAX(faiss_id)
            FROM chunks
            """
        ).fetchone()

        if row and row[0] is not None:
            return int(row[0]) + 1

        return 0

    def _save_index(self):
        if self.index is None:
            return

        faiss.write_index(self.index,self.faiss_path)

        logger.info(f"Saved FAISS index with "f"{self.index.ntotal} vectors")

    def _get_hash(self, file_path):
        try:
            with open(file_path,"rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Failed to hash {file_path}: {e}")
            return None

    def _token_count(self, text):
        if not text:
            return 0

        return len(self.encoder.tokenizer.encode(text,add_special_tokens=False))

    @staticmethod
    def _split_sentences(text):

        if not text:
            return []

        text = text.strip()

        if not text:
            return []

        sentences = re.split(r'(?<=[.!?])\s+',text)

        return [sentence.strip() for sentence in sentences if sentence.strip()]

    def _split_large_text(self, text):

        if self._token_count(text) <= self.max_tokens:

            return [text.strip()]

        sentences = self._split_sentences(text)

        if not sentences:
            sentences = [text]

        pieces = []

        current = []
        current_tokens = 0

        for sentence in sentences:

            sentence_tokens = (self._token_count(sentence))

            if sentence_tokens <= self.max_tokens:

                if (current_tokens+ sentence_tokens<= self.max_tokens):
                    current.append(sentence)
                    current_tokens += (sentence_tokens)
                else:
                    if current:
                        pieces.append(" ".join(current))
                    current = [sentence]
                    current_tokens = (sentence_tokens)

                continue

            if current:

                pieces.append(" ".join(current))

                current = []
                current_tokens = 0

            token_ids = self.encoder.tokenizer.encode(sentence,add_special_tokens=False)

            start = 0

            while start < len(token_ids):

                end = min(start + self.max_tokens,len(token_ids))

                piece_ids = token_ids[start:end]

                piece = (
                    self.encoder.tokenizer.decode(piece_ids,skip_special_tokens=True)
                )

                if piece.strip():

                    pieces.append(piece.strip())

                start = end

        if current:
            pieces.append(" ".join(current))

        return pieces

    def _build_chunks(self, text):
        
        # ----------------------------------------------------
        # Split paragraphs
        # ----------------------------------------------------

        paragraphs = re.split(r"\n\s*\n",text)

        paragraphs = [
            paragraph.strip()
            for paragraph in paragraphs
            if paragraph.strip()
        ]

        # ----------------------------------------------------
        # Convert paragraphs into manageable units
        # ----------------------------------------------------

        units = []

        for paragraph in paragraphs:

            paragraph_tokens = (self._token_count(paragraph))

            if paragraph_tokens <= self.max_tokens:

                units.append(paragraph)

            else:

                large_parts = (
                    self._split_large_text(
                        paragraph
                    )
                )

                units.extend(
                    large_parts
                )

        # ----------------------------------------------------
        # Build chunks
        # ----------------------------------------------------

        chunks = []

        current_units = []
        current_tokens = 0

        for unit in units:

            unit_tokens = (
                self._token_count(unit)
            )

            # --------------------------------------------
            # Add unit to current chunk
            # --------------------------------------------

            if (
                current_units
                and current_tokens + unit_tokens
                > self.target_tokens
            ):

                chunk_text = (
                    "\n\n".join(
                        current_units
                    )
                )

                chunks.append(
                    chunk_text
                )

                # ----------------------------------------
                # Create overlap
                # ----------------------------------------

                overlap_units = []
                overlap_count = 0

                for previous_unit in reversed(
                    current_units
                ):

                    previous_tokens = (
                        self._token_count(
                            previous_unit
                        )
                    )

                    if (
                        overlap_count
                        + previous_tokens
                        > self.overlap_tokens
                    ):
                        break

                    overlap_units.insert(
                        0,
                        previous_unit
                    )

                    overlap_count += (
                        previous_tokens
                    )

                current_units = (
                    overlap_units
                )

                current_tokens = (
                    overlap_count
                )

            # --------------------------------------------
            # Add current unit
            # --------------------------------------------

            current_units.append(
                unit
            )

            current_tokens += (
                unit_tokens
            )

        # ----------------------------------------------------
        # Final chunk
        # ----------------------------------------------------

        if current_units:

            chunks.append(
                "\n\n".join(
                    current_units
                )
            )

        return chunks


    # ============================================================
    # FILE PROCESSING
    # ============================================================

    def _process_file(self, file_path):

        topic = os.path.basename(
            os.path.dirname(file_path)
        )

        file_name = os.path.basename(
            file_path
        )

        try:

            with open(
                file_path,
                "r",
                encoding="utf-8",
                errors="ignore"
            ) as f:

                text = f.read()

        except Exception as e:

            logger.error(
                f"Failed to read "
                f"{file_path}: {e}"
            )

            return []

        if not text.strip():

            return []

        file_hash = self._get_hash(
            file_path
        )

        # ----------------------------------------------------
        # Build text chunks
        # ----------------------------------------------------

        chunk_texts = self._build_chunks(
            text
        )

        chunks = []

        # ----------------------------------------------------
        # Locate chunks in original document
        # ----------------------------------------------------

        search_start = 0

        for chunk_index, chunk_text in enumerate(
            chunk_texts
        ):

            if not chunk_text.strip():
                continue

            # ------------------------------------------------
            # Try to locate exact text
            # ------------------------------------------------

            start_pos = text.find(
                chunk_text,
                search_start
            )

            # ------------------------------------------------
            # Overlap means exact text may not be found
            # because chunks contain joined paragraph text.
            #
            # If not found, use a sequential position.
            # ------------------------------------------------

            if start_pos == -1:

                start_pos = search_start

            end_pos = (
                start_pos
                + len(chunk_text)
            )

            chunks.append({

                "file_path": file_path,
                "file_name": file_name,
                "topic": topic,
                "chunk_index": chunk_index,
                "start_pos": start_pos,
                "end_pos": end_pos,
                "chunk_text": chunk_text,
                "file_hash": file_hash
            })

            search_start = max(search_start,start_pos + 1)

        logger.info(
            f"Processed {file_name}: "f"{len(chunks)} chunks"
        )

        return chunks


    # ============================================================
    # ADD TO FAISS
    # ============================================================

    def _add_to_faiss(self, chunks):
        """
        Embed chunks and add them to FAISS.

        Embeddings are normalized so that inner product
        behaves like cosine similarity.
        """

        if not chunks:

            return []

        texts = [
            chunk["chunk_text"]
            for chunk in chunks
        ]

        embeddings = self.encoder.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        vectors = np.asarray(
            embeddings,
            dtype="float32"
        )

        ids = np.arange(
            self.next_id,
            self.next_id + len(vectors),
            dtype=np.int64
        )

        self.index.add_with_ids(
            vectors,
            ids
        )

        self.next_id += len(vectors)

        return ids.tolist()


    # ============================================================
    # REMOVE FILE FROM INDEX
    # ============================================================

    def _remove_file_from_index(self, file_path):
        """
        Remove all chunks belonging to a file from:

        1. FAISS
        2. SQLite
        """

        cursor = self.conn.cursor()

        rows = cursor.execute(
            """
            SELECT faiss_id
            FROM chunks
            WHERE file_path = ?
            """,
            (file_path,)
        ).fetchall()

        if not rows:

            return

        faiss_ids = np.array(
            [
                int(row[0])
                for row in rows
            ],
            dtype=np.int64
        )

        # ----------------------------------------------------
        # Remove from FAISS
        # ----------------------------------------------------

        try:

            if self.index is not None:

                self.index.remove_ids(
                    faiss_ids
                )

        except Exception as e:

            logger.warning(
                f"Could not remove FAISS "
                f"vectors for {file_path}: {e}"
            )

        # ----------------------------------------------------
        # Remove SQLite metadata
        # ----------------------------------------------------

        cursor.execute(
            """
            DELETE FROM chunks
            WHERE file_path = ?
            """,
            (file_path,)
        )

        self.conn.commit()

        logger.info(
            f"Removed {len(faiss_ids)} old chunks "
            f"for {os.path.basename(file_path)}"
        )


    # ============================================================
    # INDEX ONE FILE
    # ============================================================

    def _index_file(self, file_path):
        """
        Index or update one file.
        """

        with self.lock:

            if not os.path.exists(file_path):

                return

            ext = os.path.splitext(
                file_path
            )[1].lower()

            if ext not in [
                ".txt",
                ".md"
            ]:

                return

            file_hash = self._get_hash(
                file_path
            )

            if not file_hash:

                return

            # ------------------------------------------------
            # Check existing file
            # ------------------------------------------------

            cursor = self.conn.cursor()

            existing = cursor.execute(
                """
                SELECT faiss_id, file_hash
                FROM chunks
                WHERE file_path = ?
                """,
                (file_path,)
            ).fetchall()

            # ------------------------------------------------
            # Existing and unchanged
            # ------------------------------------------------

            if existing:

                old_hash = existing[0][1]

                if old_hash == file_hash:

                    logger.info(
                        f"Skipping unchanged: "
                        f"{os.path.basename(file_path)}"
                    )

                    return

                # ------------------------------------------------
                # File changed
                # ------------------------------------------------

                logger.info(
                    f"File changed: "
                    f"{os.path.basename(file_path)}"
                )

                self._remove_file_from_index(
                    file_path
                )

            # ------------------------------------------------
            # Process file
            # ------------------------------------------------

            chunks = self._process_file(
                file_path
            )

            if not chunks:

                logger.warning(
                    f"No chunks generated for "
                    f"{file_path}"
                )

                return

            # ------------------------------------------------
            # Add vectors
            # ------------------------------------------------

            faiss_ids = self._add_to_faiss(
                chunks
            )

            # ------------------------------------------------
            # Store metadata
            # ------------------------------------------------

            for chunk, faiss_id in zip(
                chunks,
                faiss_ids
            ):

                cursor.execute(
                    """
                    INSERT INTO chunks (
                        faiss_id,
                        file_path,
                        file_name,
                        topic,
                        chunk_index,
                        start_pos,
                        end_pos,
                        chunk_text,
                        file_hash
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        faiss_id,
                        chunk["file_path"],
                        chunk["file_name"],
                        chunk["topic"],
                        chunk["chunk_index"],
                        chunk["start_pos"],
                        chunk["end_pos"],
                        chunk["chunk_text"],
                        chunk["file_hash"]
                    )
                )

            self.conn.commit()

            self._save_index()

            logger.info(
                f"Indexed: "
                f"{os.path.basename(file_path)} "
                f"({len(chunks)} chunks)"
            )


    # ============================================================
    # INDEX ALL
    # ============================================================

    def _index_all(self):
        """
        Index all current documents.
        """

        logger.info(
            "Indexing all files..."
        )

        current_files = set()

        # ----------------------------------------------------
        # Find current files
        # ----------------------------------------------------

        for root, _, files in os.walk(
            self.docs_dir
        ):

            for file_name in files:

                file_path = os.path.join(
                    root,
                    file_name
                )

                ext = os.path.splitext(
                    file_path
                )[1].lower()

                if ext not in [
                    ".txt",
                    ".md"
                ]:
                    continue

                current_files.add(
                    os.path.abspath(
                        file_path
                    )
                )

        # ----------------------------------------------------
        # Remove database entries for files that no longer exist
        # ----------------------------------------------------

        db_files = self.conn.execute(
            """
            SELECT DISTINCT file_path
            FROM chunks
            """
        ).fetchall()

        for row in db_files:

            db_file = os.path.abspath(
                row[0]
            )

            if db_file not in current_files:

                logger.info(
                    f"Removing deleted file: "
                    f"{db_file}"
                )

                self._remove_file_from_index(
                    db_file
                )

        # ----------------------------------------------------
        # Index current files
        # ----------------------------------------------------

        for file_path in current_files:

            self._index_file(
                file_path
            )

        self._save_index()

        count = self.conn.execute(
            """
            SELECT COUNT(*)
            FROM chunks
            """
        ).fetchone()[0]

        logger.info(
            f"Indexing complete: "
            f"{count} chunks"
        )


    # ============================================================
    # SEARCH
    # ============================================================

    def search(
        self,
        query,
        topic=None,
        top_k=5
    ):
        """
        Hybrid search:

        1. FAISS semantic similarity
        2. Exact phrase matching
        3. Keyword matching
        4. Combined ranking
        """

        with self.lock:

            if not query:

                return []

            query = query.strip()

            if not query:

                return []

            if self.index is None:

                return []

            if self.index.ntotal == 0:

                return []

            # ------------------------------------------------
            # Query embedding
            # ------------------------------------------------

            query_embedding = self.encoder.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            ).astype(
                "float32"
            )

            # ------------------------------------------------
            # Search more candidates
            # ------------------------------------------------

            search_k = min(
                max(
                    top_k * 10,
                    50
                ),
                self.index.ntotal
            )

            scores, indices = (
                self.index.search(
                    query_embedding,
                    search_k
                )
            )

            if (
                len(indices) == 0
                or len(indices[0]) == 0
            ):

                return []

            faiss_ids = [int(idx) for idx in indices[0] if idx >= 0]

            if not faiss_ids:
                return []

            placeholders = ",".join(["?"] * len(faiss_ids))

            cursor = self.conn.cursor()

            if topic:
                rows = cursor.execute(
                    f"""
                    SELECT
                        faiss_id,
                        file_path,
                        file_name,
                        topic,
                        chunk_index,
                        start_pos,
                        end_pos,
                        chunk_text
                    FROM chunks
                    WHERE faiss_id IN ({placeholders})
                    AND topic = ?
                    """,
                    faiss_ids + [topic]
                ).fetchall()

            else:
                rows = cursor.execute(
                    f"""
                    SELECT
                        faiss_id,
                        file_path,
                        file_name,
                        topic,
                        chunk_index,
                        start_pos,
                        end_pos,
                        chunk_text
                    FROM chunks
                    WHERE faiss_id IN ({placeholders})
                    """,
                    faiss_ids
                ).fetchall()

            row_by_id = {int(row[0]): row for row in rows}

            query_lower = query.lower()

            query_words = [word for word in re.findall(r"\b\w+\b",query_lower)if len(word) > 1]

            candidates = []

            for rank, (faiss_id,similarity) in enumerate(zip(indices[0],scores[0])):

                faiss_id = int(faiss_id)

                if faiss_id < 0:
                    continue

                if faiss_id not in row_by_id:
                    continue

                row = row_by_id[faiss_id]

                text = row[7]

                text_lower = (text.lower())

                semantic_score = float(similarity)

                semantic_score = (semantic_score + 1.0) / 2.0
                semantic_score = max(
                    0.0,min(1.0,semantic_score)
                )
                exact_phrase_score = 0.0
                if query_lower in text_lower:
                    exact_phrase_score = 1.0
                if query_words:
                    matched_words = sum(1 for word in query_words if word in text_lower)
                    keyword_score = (matched_words/ len(query_words))
                else:
                    keyword_score = 0.0
                final_score = (
                    semantic_score * 1.0+exact_phrase_score * 5.0+keyword_score * 2.0
                )

                candidates.append({
                    "score": final_score,
                    "semantic_score":semantic_score,
                    "exact_phrase_score":exact_phrase_score,
                    "keyword_score":keyword_score,
                    "faiss_similarity":float(similarity),
                    "faiss_rank":rank,
                    "row":row
                })
            candidates.sort(
                key=lambda x: (x["score"],x["semantic_score"]),reverse=True
            )
            results = []
            for candidate in candidates[:top_k]:
                row = candidate["row"]
                results.append({
                    "file":row[2],
                    "file_path":row[1],
                    "topic":row[3],
                    "chunk_index":row[4],
                    "start_pos":row[5],
                    "end_pos":row[6],
                    "text":row[7],
                    "score":round(candidate["score"],4),
                    "semantic_score":round(candidate["semantic_score"],4),
                    "exact_match":(candidate["exact_phrase_score"] > 0),
                    "keyword_score":round(candidate["keyword_score"],4),
                    "faiss_similarity":round(candidate["faiss_similarity"],4)
                })

            return results

    def search_all(self,query,top_k=5):
        return self.search(query=query,topic=None,top_k=top_k)


    def search_topic(self,query,topic,top_k=5):
        return self.search(query=query,topic=topic,top_k=top_k)

    def stats(self):
        """
        Return system statistics.
        """

        count = self.conn.execute(
            """
            SELECT COUNT(*)
            FROM chunks
            """
        ).fetchone()[0]

        files = self.conn.execute(
            """
            SELECT COUNT(
                DISTINCT file_path
            )
            FROM chunks
            """
        ).fetchone()[0]

        topics = self.conn.execute(
            """
            SELECT
                topic,
                COUNT(*) as count
            FROM chunks
            GROUP BY topic
            """
        ).fetchall()

        return {

            "chunks":
                count,

            "files":
                files,

            "faiss_vectors":
                (
                    self.index.ntotal
                    if self.index
                    else 0
                ),

            "embedding_model":
                self.model_name,

            "embedding_dimension":
                self.dimension,

            "target_tokens":
                self.target_tokens,

            "max_tokens":
                self.max_tokens,

            "overlap_tokens":
                self.overlap_tokens,

            "topics": [
                {
                    "topic": topic,
                    "chunks": count
                }
                for topic, count in topics
            ],

            "watching":
                (
                    self.observer.is_alive()
                    if self.observer
                    else False
                )
        }

    def reindex_all(self):
        with self.lock:

            logger.info("Starting full reindex...")

            self.conn.execute("DELETE FROM chunks")

            self.conn.commit()

            self._create_empty_index()

            for root, _, files in os.walk(self.docs_dir):

                for file_name in files:

                    file_path = os.path.join(root,file_name)

                    ext = os.path.splitext(file_path)[1].lower()

                    if ext not in [".txt",".md"]:
                        continue
                    self._index_file(file_path)
            self._save_index()

            count = self.conn.execute(
                """
                SELECT COUNT(*)
                FROM chunks
                """
            ).fetchone()[0]

            logger.info(
                f"Full reindex complete: "
                f"{count} chunks"
            )

    def _start_watcher(self):

        system = self

        class Handler(FileSystemEventHandler):

            def on_created(self,event):
                if event.is_directory:
                    return

                logger.info(
                    f"New file: "
                    f"{os.path.basename(event.src_path)}"
                )

                system._index_file(event.src_path)


            def on_modified(self,event):
                if event.is_directory:
                    return

                logger.info(
                    f"Modified file: "
                    f"{os.path.basename(event.src_path)}"
                )
                system._index_file(event.src_path)


            def on_deleted(self,event):
                if event.is_directory:
                    return
                
                logger.info(
                    f"Deleted file: "
                    f"{os.path.basename(event.src_path)}"
                )

                with system.lock:
                    system._remove_file_from_index(event.src_path)
                    system._save_index()
        self.observer = Observer()
        self.observer.schedule(Handler(),self.docs_dir,recursive=True)
        self.observer.start()
        logger.info(f"Watching: "f"{self.docs_dir}"
        )

    def stop(self):
        with self.lock:
            if self.observer:
                self.observer.stop()
                self.observer.join()
                self.observer = None
            self._save_index()
            if self.conn:
                self.conn.close()
                self.conn = None
            logger.info(
                "RAG system stopped"
            )


