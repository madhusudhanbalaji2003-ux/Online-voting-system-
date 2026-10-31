"""
Blockchain module for SecureVote Online Voting System.
Implements a simple blockchain for immutable vote storage.
Votes are persisted to SQLite so they survive serverless cold starts.
"""

import hashlib
import json
import os
import sqlite3
import time


# ── Persistence helpers ────────────────────────────────────────────────────────
_IS_VERCEL = bool(os.environ.get("VERCEL"))
_CHAIN_DB  = "/tmp/blockchain.db" if _IS_VERCEL else os.path.join(os.path.dirname(__file__), "blockchain_chain.db")


def _get_chain_conn():
    conn = sqlite3.connect(_CHAIN_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS blocks (
            idx       INTEGER PRIMARY KEY,
            timestamp REAL,
            data      TEXT,
            prev_hash TEXT,
            nonce     INTEGER,
            hash      TEXT
        )
    """)
    conn.commit()
    return conn


class Block:
    """Represents a single block in the blockchain."""

    def __init__(self, index, timestamp, data, previous_hash, nonce=0, hash_val=None):
        self.index         = index
        self.timestamp     = timestamp
        self.data          = data
        self.previous_hash = previous_hash
        self.nonce         = nonce
        self.hash          = hash_val if hash_val else self.calculate_hash()

    def calculate_hash(self):
        """Calculate SHA-256 hash of the block."""
        block_string = json.dumps({
            "index":         self.index,
            "timestamp":     self.timestamp,
            "data":          self.data,
            "previous_hash": self.previous_hash,
            "nonce":         self.nonce
        }, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()

    def mine_block(self, difficulty=2):
        """Simple proof-of-work mining."""
        target = "0" * difficulty
        while self.hash[:difficulty] != target:
            self.nonce += 1
            self.hash   = self.calculate_hash()

    def to_dict(self):
        """Convert block to dictionary for API responses."""
        return {
            "index":         self.index,
            "timestamp":     self.timestamp,
            "data":          self.data,
            "previous_hash": self.previous_hash,
            "nonce":         self.nonce,
            "hash":          self.hash
        }


class Blockchain:
    """Blockchain for storing votes immutably, persisted to SQLite."""

    def __init__(self):
        self.difficulty = 2
        self.chain      = self._load_chain()

    # ── Persistence ─────────────────────────────────────────────────────────
    def _load_chain(self):
        """Load chain from SQLite; create genesis block if empty."""
        try:
            conn   = _get_chain_conn()
            rows   = conn.execute("SELECT idx, timestamp, data, prev_hash, nonce, hash FROM blocks ORDER BY idx").fetchall()
            conn.close()
            if rows:
                chain = []
                for row in rows:
                    b = Block(
                        index         = row[0],
                        timestamp     = row[1],
                        data          = json.loads(row[2]),
                        previous_hash = row[3],
                        nonce         = row[4],
                        hash_val      = row[5]
                    )
                    chain.append(b)
                return chain
        except Exception as e:
            print(f"[Blockchain] Failed to load chain: {e}")

        # Fresh chain
        genesis = Block(0, time.time(), {"message": "Genesis Block - SecureVote Initialized"}, "0")
        genesis.mine_block(self.difficulty)
        self._save_block(genesis)
        return [genesis]

    def _save_block(self, block):
        """Persist a single block to SQLite."""
        try:
            conn = _get_chain_conn()
            conn.execute(
                "INSERT OR REPLACE INTO blocks (idx, timestamp, data, prev_hash, nonce, hash) VALUES (?, ?, ?, ?, ?, ?)",
                (block.index, block.timestamp, json.dumps(block.data), block.previous_hash, block.nonce, block.hash)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[Blockchain] Failed to save block: {e}")

    def _clear_chain_db(self):
        """Delete all blocks from the persistence layer."""
        try:
            conn = _get_chain_conn()
            conn.execute("DELETE FROM blocks")
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[Blockchain] Failed to clear chain db: {e}")

    # ── Public API ──────────────────────────────────────────────────────────
    def create_genesis_block(self):
        """Create the first block in the chain."""
        genesis = Block(0, time.time(), {"message": "Genesis Block - SecureVote Initialized"}, "0")
        genesis.mine_block(self.difficulty)
        self.chain.append(genesis)
        self._save_block(genesis)

    def get_latest_block(self):
        """Get the most recent block."""
        return self.chain[-1]

    def add_vote(self, voter_id, candidate_id, candidate_name, election_id):
        """
        Add a vote as a new block in the blockchain.
        Returns the new block.
        """
        data = {
            "type":           "vote",
            "voter_id_hash":  hashlib.sha256(voter_id.encode()).hexdigest(),
            "candidate_id":   candidate_id,
            "candidate_name": candidate_name,
            "election_id":    election_id,
            "timestamp":      time.time()
        }

        new_block = Block(
            index         = len(self.chain),
            timestamp     = time.time(),
            data          = data,
            previous_hash = self.get_latest_block().hash
        )
        new_block.mine_block(self.difficulty)
        self.chain.append(new_block)
        self._save_block(new_block)
        return new_block

    def is_chain_valid(self):
        """Validate the integrity of the entire blockchain."""
        for i in range(1, len(self.chain)):
            current  = self.chain[i]
            previous = self.chain[i - 1]

            if current.hash != current.calculate_hash():
                return False

            if current.previous_hash != previous.hash:
                return False

        return True

    def get_chain(self):
        """Return the full chain as a list of dictionaries."""
        return [block.to_dict() for block in self.chain]

    def get_vote_counts(self, election_id=None):
        """Tally votes from the blockchain."""
        counts = {}
        for block in self.chain:
            if block.data.get("type") == "vote":
                if election_id and block.data.get("election_id") != election_id:
                    continue
                candidate = block.data["candidate_name"]
                counts[candidate] = counts.get(candidate, 0) + 1
        return counts

    def get_total_votes(self):
        """Get total number of votes cast."""
        return sum(1 for block in self.chain if block.data.get("type") == "vote")

    def reset(self):
        """Reset blockchain (admin only — for new elections)."""
        self._clear_chain_db()
        self.chain = []
        self.create_genesis_block()
