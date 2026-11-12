"""
Blockchain module for SecureVote Online Voting System.
Implements a simple blockchain for immutable vote storage.

Performance notes
─────────────────
• get_vote_counts()   – result is cached in _vote_cache; invalidated on add_vote()
• is_chain_valid()    – result is cached in _valid_cache; invalidated on add_vote()
• mine_block()        – difficulty=2 (unchanged); cache prevents re-validation overhead
"""

import hashlib
import json
import time


class Block:
    """Represents a single block in the blockchain."""

    def __init__(self, index, timestamp, data, previous_hash):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.nonce = 0
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        """Calculate SHA-256 hash of the block."""
        block_string = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "data": self.data,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce
        }, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()

    def mine_block(self, difficulty=2):
        """Simple proof-of-work mining."""
        target = "0" * difficulty
        while self.hash[:difficulty] != target:
            self.nonce += 1
            self.hash = self.calculate_hash()

    def to_dict(self):
        """Convert block to dictionary for API responses."""
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "data": self.data,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce,
            "hash": self.hash
        }


class Blockchain:
    """Blockchain for storing votes immutably."""

    def __init__(self):
        self.chain = []
        self.difficulty = 2
        # ── Performance caches ──────────────────────────────────────
        # Invalidated every time a new block is added.
        self._vote_cache: dict = {}          # election_id -> {candidate: count}
        self._valid_cache: bool | None = None  # True/False or None (not computed)
        # ────────────────────────────────────────────────────────────
        self.create_genesis_block()

    def _invalidate_caches(self):
        """Clear cached results whenever the chain changes."""
        self._vote_cache.clear()
        self._valid_cache = None

    def create_genesis_block(self):
        """Create the first block in the chain."""
        genesis = Block(0, time.time(), {"message": "Genesis Block - SecureVote Initialized"}, "0")
        genesis.mine_block(self.difficulty)
        self.chain.append(genesis)
        self._invalidate_caches()

    def get_latest_block(self):
        """Get the most recent block."""
        return self.chain[-1]

    def add_vote(self, voter_id, candidate_id, candidate_name, election_id):
        """
        Add a vote as a new block in the blockchain.
        Returns the new block.
        """
        data = {
            "type": "vote",
            "voter_id_hash": hashlib.sha256(voter_id.encode()).hexdigest(),
            "candidate_id": candidate_id,
            "candidate_name": candidate_name,
            "election_id": election_id,
            "timestamp": time.time()
        }

        new_block = Block(
            index=len(self.chain),
            timestamp=time.time(),
            data=data,
            previous_hash=self.get_latest_block().hash
        )
        new_block.mine_block(self.difficulty)
        self.chain.append(new_block)
        self._invalidate_caches()   # ← invalidate so next read is fresh
        return new_block

    def is_chain_valid(self):
        """Validate the integrity of the entire blockchain.

        Result is cached after the first call and reused until a new block
        is added, avoiding an O(n) hash-recalculation on every page load.
        """
        if self._valid_cache is not None:
            return self._valid_cache

        for i in range(1, len(self.chain)):
            current  = self.chain[i]
            previous = self.chain[i - 1]

            if current.hash != current.calculate_hash():
                self._valid_cache = False
                return False

            if current.previous_hash != previous.hash:
                self._valid_cache = False
                return False

        self._valid_cache = True
        return True

    def get_chain(self):
        """Return the full chain as a list of dictionaries."""
        return [block.to_dict() for block in self.chain]

    def get_vote_counts(self, election_id=None):
        """Tally votes from the blockchain.

        Results are cached per election_id to avoid scanning the full chain
        on every index / results / admin page load.
        """
        cache_key = election_id if election_id is not None else "__all__"
        if cache_key in self._vote_cache:
            return dict(self._vote_cache[cache_key])  # return a copy

        counts: dict = {}
        for block in self.chain:
            if block.data.get("type") == "vote":
                if election_id and block.data.get("election_id") != election_id:
                    continue
                candidate = block.data["candidate_name"]
                counts[candidate] = counts.get(candidate, 0) + 1

        self._vote_cache[cache_key] = counts
        return dict(counts)

    def get_total_votes(self):
        """Get total number of votes cast."""
        return sum(1 for block in self.chain if block.data.get("type") == "vote")

    def reset(self):
        """Reset blockchain (admin only — for new elections)."""
        self.chain = []
        self._invalidate_caches()
        self.create_genesis_block()

