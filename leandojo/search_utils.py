# Snippets from https://github.com/lean-dojo/ReProver

from dataclasses import dataclass, field
from enum import Enum
from functools import total_ordering
import heapq
import json
from loguru import logger
import networkx as nx
import re
import time
import torch
from typing import Optional, List, Dict, Any, Tuple, Generator

from lean_dojo import (
    Pos,
    Dojo,
    Theorem,
    LeanGitRepo,
    TacticState,
    LeanError,
    TimeoutError,
    ProofFinished,
    ProofGivenUp,
    DojoInitError,
    DojoCrashError,
    DojoHardTimeoutError,
)

from search_tree import *

Example = Dict[str, Any]
Batch = Dict[str, Any]

MARK_START_SYMBOL = "<a>"
MARK_END_SYMBOL = "</a>"


def remove_marks(s: str) -> str:
    """Remove all :code:`<a>` and :code:`</a>` from ``s``."""
    return s.replace(MARK_START_SYMBOL, "").replace(MARK_END_SYMBOL, "")


@dataclass(unsafe_hash=True)
class Context:
    """Contexts are "queries" in our retrieval setup."""

    path: str
    theorem_full_name: str
    theorem_pos: Pos = field(compare=False)
    state: str

    def __post_init__(self) -> None:
        assert isinstance(self.path, str)
        assert isinstance(self.theorem_full_name, str)
        assert isinstance(self.theorem_pos, Pos)
        assert (
            isinstance(self.state, str)
            and "⊢" in self.state
            and MARK_START_SYMBOL not in self.state
            and MARK_END_SYMBOL not in self.state
        )

    def serialize(self) -> str:
        """Serialize the context into a string for Transformers."""
        return self.state


@dataclass(unsafe_hash=True)
class Premise:
    """Premises are "documents" in our retrieval setup."""

    path: str
    """The ``*.lean`` file this premise comes from.
    """

    full_name: str
    """Fully qualified name.
    """

    start: Pos = field(repr=False)
    """Start position of the premise's definition in the ``*.lean`` file.
    """

    end: Pos = field(repr=False, compare=False)
    """End position of the premise's definition in the ``*.lean`` file.
    """

    code: str = field(compare=False)
    """Raw, human-written code for defining the premise.
    """

    def __post_init__(self) -> None:
        assert isinstance(self.path, str)
        assert isinstance(self.full_name, str)
        assert (
            isinstance(self.start, Pos)
            and isinstance(self.end, Pos)
            and self.start <= self.end
        )
        assert isinstance(self.code, str) and self.code != ""

    def serialize(self) -> str:
        """Serialize the premise into a string for Transformers."""
        annot_full_name = f"{MARK_START_SYMBOL}{self.full_name}{MARK_END_SYMBOL}"
        code = self.code.replace(f"_root_.{self.full_name}", annot_full_name)
        fields = self.full_name.split(".")

        for i in range(len(fields)):
            prefix = ".".join(fields[i:])
            new_code = re.sub(f"(?<=\s)«?{prefix}»?", annot_full_name, code)
            if new_code != code:
                code = new_code
                break

        return code


class PremiseSet:
    """A set of premises indexed by their paths and full names."""

    path2premises: Dict[str, Dict[str, Premise]]

    def __init__(self) -> None:
        self.path2premises = {}

    def __iter__(self) -> Generator[Premise, None, None]:
        for _, premises in self.path2premises.items():
            for p in premises.values():
                yield p

    def add(self, p: Premise) -> None:
        if p.path in self.path2premises:
            self.path2premises[p.path][p.full_name] = p
        else:
            self.path2premises[p.path] = {p.full_name: p}

    def update(self, premises: List[Premise]) -> None:
        for p in premises:
            self.add(p)

    def __contains__(self, p: Premise) -> bool:
        return (
            p.path in self.path2premises and p.full_name in self.path2premises[p.path]
        )

    def __len__(self) -> int:
        return sum(len(premises) for premises in self.path2premises.values())


@dataclass(frozen=True)
class File:
    """A file defines 0 or multiple premises."""

    path: str
    """Path of the ``*.lean`` file.
    """

    premises: List[Premise] = field(repr=False, compare=False)
    """A list of premises defined in this file.
    """

    @classmethod
    def from_data(cls, file_data: Dict[str, Any]) -> "File":
        """Construct a :class:`File` object from ``file_data``."""
        path = file_data["path"]
        premises = []
        for p in file_data["premises"]:
            full_name = p["full_name"]
            if full_name is None:
                continue
            if "user__.n" in full_name or p["code"] == "":
                # Ignore ill-formed premises (often due to errors in ASTs).
                continue
            if full_name.startswith("[") and full_name.endswith("]"):
                # Ignore mutual definitions.
                continue
            premises.append(
                Premise(
                    path, p["full_name"], Pos(*p["start"]), Pos(*p["end"]), p["code"]
                )
            )
        return cls(path, premises)

    @property
    def is_empty(self) -> bool:
        """Check whether the file contains no premise."""
        return self.premises == []


class Corpus:
    """Our retrieval corpus is a DAG of files. Each file consists of
    premises (theorems, definitoins, etc.) that can be retrieved.
    """

    transitive_dep_graph: nx.DiGraph
    """Transitive closure of the dependency graph among files. 
    There is an edge from file X to Y iff X import Y (directly or indirectly).
    """

    all_premises: List[Premise]
    """All premises in the entire corpus.
    """

    def __init__(self, jsonl_path: str) -> None:
        """Construct a :class:`Corpus` object from a ``corpus.jsonl`` data file."""
        dep_graph = nx.DiGraph()
        self.all_premises = []

        logger.info(f"Building the corpus from {jsonl_path}")

        for line in open(jsonl_path):
            file_data = json.loads(line)
            path = file_data["path"]
            assert not dep_graph.has_node(path)
            file = File.from_data(file_data)

            dep_graph.add_node(path, file=file)
            self.all_premises.extend(file.premises)

            for p in file_data["imports"]:
                assert dep_graph.has_node(p)
                dep_graph.add_edge(path, p)

        assert nx.is_directed_acyclic_graph(dep_graph)
        self.transitive_dep_graph = nx.transitive_closure_dag(dep_graph)

        self.imported_premises_cache = {}
        self.fill_cache()

    def _get_file(self, path: str) -> File:
        return self.transitive_dep_graph.nodes[path]["file"]

    def __len__(self) -> int:
        return len(self.all_premises)

    def __contains__(self, path: str) -> bool:
        return path in self.transitive_dep_graph

    def __getitem__(self, idx: int) -> Premise:
        return self.all_premises[idx]

    @property
    def files(self) -> List[File]:
        return [self._get_file(p) for p in self.transitive_dep_graph.nodes]

    @property
    def num_files(self) -> int:
        return len(self.files)

    def get_dependencies(self, path: str) -> List[str]:
        """Return a list of (direct and indirect) dependencies of the file ``path``."""
        return list(self.transitive_dep_graph.successors(path))

    def get_premises(self, path: str) -> List[Premise]:
        """Return a list of premises defined in the file ``path``."""
        return self._get_file(path).premises

    def num_premises(self, path: str) -> int:
        """Return the number of premises defined in the file ``path``."""
        return len(self.get_premises(path))

    def locate_premise(self, path: str, pos: Pos) -> Optional[Premise]:
        """Return a premise at position ``pos`` in file ``path``.

        Return None if no such premise can be found.
        """
        for p in self.get_premises(path):
            assert p.path == path
            if p.start <= pos <= p.end:
                return p
        return None

    def fill_cache(self) -> None:
        for path in self.transitive_dep_graph.nodes:
            self._get_imported_premises(path)

    def _get_imported_premises(self, path: str) -> List[Premise]:
        """Return a list of premises imported in file ``path``. The result is cached."""
        premises = self.imported_premises_cache.get(path, None)
        if premises is not None:
            return premises

        premises = []
        for p in self.transitive_dep_graph.successors(path):
            premises.extend(self._get_file(p).premises)
        self.imported_premises_cache[path] = premises
        return premises

    def get_accessible_premises(self, path: str, pos: Pos) -> PremiseSet:
        """Return the set of premises accessible at position ``pos`` in file ``path``,
        i.e., all premises defined in the (transitively) imported files or earlier in the same file.
        """
        premises = PremiseSet()
        for p in self.get_premises(path):
            if p.end <= pos:
                premises.add(p)
        premises.update(self._get_imported_premises(path))
        return premises

    def get_accessible_premise_indexes(self, path: str, pos: Pos) -> List[int]:
        return [
            i
            for i, p in enumerate(self.all_premises)
            if (p.path == path and p.end <= pos)
            or self.transitive_dep_graph.has_edge(path, p.path)
        ]

    def get_nearest_premises(
        self,
        premise_embeddings: torch.FloatTensor,
        batch_context: List[Context],
        batch_context_emb: torch.Tensor,
        k: int,
    ) -> Tuple[List[List[Premise]], List[List[float]]]:
        """Perform a batch of nearest neighbour search."""
        similarities = batch_context_emb @ premise_embeddings.t()
        idxs_batch = similarities.argsort(dim=1, descending=True).tolist()
        results = [[] for _ in batch_context]
        scores = [[] for _ in batch_context]

        for j, (ctx, idxs) in enumerate(zip(batch_context, idxs_batch)):
            accessible_premises = self.get_accessible_premises(
                ctx.path, ctx.theorem_pos
            )
            for i in idxs:
                p = self.all_premises[i]
                if p in accessible_premises:
                    results[j].append(p)
                    scores[j].append(similarities[j, i].item())
                    if len(results[j]) >= k:
                        break
            else:
                raise ValueError

        return results, scores
        

@dataclass(frozen=True)
class SearchResult:
    """The result of attempting to prove a theorem."""

    theorem: Theorem
    status: Status
    proof: Optional[List[str]]

    # Some statistics during proof search.
    actor_time: float
    environment_time: float
    total_time: float
    num_total_nodes: int
    num_searched_nodes: int


class BestFirstSearchProver:
    """A prover that uses best-first search to find proofs using a tactic generator."""

    def __init__(
        self,
        tac_gen,  # A given tactic generator.
        timeout: int,
        num_sampled_tactics: int,
        debug: bool,
    ) -> None:
        self.tac_gen = tac_gen
        self.timeout = timeout
        self.num_sampled_tactics = num_sampled_tactics
        self.debug = debug

        self.num_expansions = 0
        self.actor_time = 0.0
        self.environment_time = 0.0
        self.total_time = None

    def search(
        self, repo: LeanGitRepo, thm: Theorem, pos: Pos
    ) -> Optional[SearchResult]:
        logger.info(f"Proving {thm}")

        self.repo = repo
        self.theorem = thm
        self.posision = pos
        self.actor_time = 0.0
        self.environment_time = 0.0
        self.num_expansions = 0

        try:
            with Dojo(thm, hard_timeout=60 + self.timeout) as (
                dojo,
                init_state,
            ):
                self.dojo = dojo
                self.root = InternalNode(
                    state=init_state,
                    cumulative_logprob=0.0,
                )
                self.nodes = {init_state: self.root}
                self.priority_queue = [self.root]

                with torch.no_grad():
                    try:
                        self._best_first_search()
                    except DojoCrashError as ex:
                        logger.warning(f"Dojo crashed with {ex} when proving {thm}")
                        pass

            if self.root.status == Status.PROVED:
                proof = [e.tactic for e in self.root.extract_proof()]
            else:
                proof = None

            result = SearchResult(
                theorem=thm,
                status=self.root.status,
                proof=proof,
                actor_time=self.actor_time,
                environment_time=self.environment_time,
                total_time=self.total_time,
                num_total_nodes=len(self.nodes),
                num_searched_nodes=self.num_expansions,
            )
            logger.info(result)
            return result

        except DojoInitError as ex:
            logger.warning(ex)
            return None

    def _best_first_search(self) -> None:
        time_start = time.monotonic()

        while True:
            if len(self.priority_queue) == 0:
                logger.info("Ran out of nodes to search.")
                break

            try:
                self._step()
            except DojoHardTimeoutError:
                assert time.monotonic() - time_start >= self.timeout

            self.total_time = time.monotonic() - time_start
            if self.total_time > self.timeout:
                if self.root.status == Status.PROVED:
                    logger.info("Found a proof but timed out.")
                self.root.status = Status.OPEN
                logger.info("Search timed out.")
                break

            if self.root.status == Status.FAILED:
                logger.info("Failed early!")
                break

            if self.root.status == Status.PROVED:
                logger.info("Found a proof!")
                break

    def _step(self):
        """
        Perform a single step of search.

        Selects the node with the highest priority, queries the model for suggested
        tactics, and tries each tactic in the environment, creating and enqueuing
        a new node for each valid result.
        """
        # Search the node with highest priority.
        search_node = heapq.heappop(self.priority_queue)
        logger.debug(f"Expanding node: {search_node}")

        if self.debug:
            assert all(
                search_node.priority >= node.priority for node in self.priority_queue
            )

        if isinstance(search_node.state, TacticState):
            ts = search_node.state.pp
        else:
            ts = search_node.state.unsolved_tactic_state
        suggestions = self._generate_tactics(ts)

        # Try all tactics in order of descending logprob, and collect the results. Any
        # new nodes are added to `self.nodes`, and edges are added to the result node.
        results = []
        for tactic, logprob in suggestions:
            edge, finished = self._run_tactic(search_node, tactic, logprob)
            results.append(edge)
            if finished:
                break

        # Store the fixed out edges of this node, marking it as explored.
        # This will trigger recursively recomputing tree statistics.
        search_node.out_edges = results
        self.num_expansions += 1

        # If we're running in debug mode, run a full test suite each step
        if self.debug:
            assert self.num_expansions == sum(
                node.is_explored
                for node in self.nodes.values()
                if isinstance(node, InternalNode)
            )
            self.check_invariants()

    def _generate_tactics(self, ts: str) -> List[Tuple[str, float]]:
        t0 = time.monotonic()

        path = str(self.theorem.file_path)

        suggestions = self.tac_gen.generate(
            state=ts,
            file_path=path,
            theorem_full_name=self.theorem.full_name,
            theorem_pos=self.posision,
            num_samples=self.num_sampled_tactics,
        )

        self.actor_time += time.monotonic() - t0

        logger.debug(f"Tactic suggestions: {suggestions}")
        return suggestions

    def _run_tactic(
        self, node: InternalNode, tactic: str, logprob: float
    ) -> Tuple[Edge, bool]:
        t0 = time.monotonic()
        response = self.dojo.run_tac(node.state, tactic)

        elapsed = time.monotonic() - t0
        self.environment_time += elapsed

        try:
            # If we've seen this response before, use the existing node
            result_node = self.nodes[response]
        except KeyError:
            # Build a new node
            if isinstance(response, ProofFinished):
                result_node = ProofFinishedNode(response)
            elif type(response) in (
                LeanError,
                TimeoutError,
                ProofGivenUp,
            ):
                result_node = ErrorNode(response)
            else:
                assert isinstance(response, TacticState)
                result_node = InternalNode(
                    state=response,
                    cumulative_logprob=logprob + node.cumulative_logprob,
                )
            if result_node.status == Status.OPEN:  # Don't search proved/failed nodes
                heapq.heappush(self.priority_queue, result_node)  # type: ignore

        # Record the new node and add it to the search queue.
        self.nodes[response] = result_node

        # Build an edge connecting these nodes.
        # Will be added to the source node externally.
        edge = Edge(tactic=tactic, src=node, dst=result_node)

        if isinstance(result_node, InternalNode):
            result_node.in_edges.append(edge)

        return edge, isinstance(response, ProofFinished)

