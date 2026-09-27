"""
Condition nodes for Behavior Trees grounded in PDDL specifications.
Evaluates whether problem goal predicates or preconditions hold in the environment.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import py_trees

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
CONDITION_REGISTRY: Dict[str, type[py_trees.behaviour.Behaviour]] = {}


def register_condition(name: str):
    """Decorator to register condition classes in CONDITION_REGISTRY."""
    def decorator(cls: type[py_trees.behaviour.Behaviour]):
        CONDITION_REGISTRY[name] = cls
        return cls
    return decorator


# ---------------------------------------------------------------------------
# PDDL Problem Goal Extraction
# ---------------------------------------------------------------------------
def extract_goals_from_problem(problem_source: Optional[str] = None) -> List[Tuple[str, ...]]:
    """
    Extract atomic goal predicates from a problem.pddl file or raw string.

    Returns:
        e.g. [('on', 'red_can', 'sorting_bin'), ('on', 'yellow_cube', 'pot'), ...]
    """
    if problem_source is None:
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(curr_dir, "..", ".."))
        candidates = [
            os.path.join(repo_root, "experiments", "generated_problems", "problem.pddl"),
            os.path.join(repo_root, "generated_problems", "problem.pddl"),
            os.path.join(repo_root, "problem.pddl"),
        ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                problem_source = candidate
                break

    if not problem_source:
        return []

    text = ""
    if os.path.isfile(str(problem_source)):
        with open(problem_source, "r") as f:
            text = f.read()
    else:
        text = str(problem_source)

    # Remove comments
    text = re.sub(r";.*", "", text)

    if "(:goal" not in text:
        return []

    goal_part = text.split("(:goal")[1]
    atoms = re.findall(r"\(([a-zA-Z0-9_\-]+)\s+([a-zA-Z0-9_\-\s]+)\)", goal_part)
    goals = []
    for pred, args_str in atoms:
        if pred.lower() in ("and", "not", "or", "imply"):
            continue
        args = args_str.split()
        goals.append((pred, *args))

    return goals


def evaluate_pddl_predicate(env: Any, predicate: str, args: List[str]) -> bool:
    """
    Evaluate whether a PDDL predicate holds true in the simulation environment.
    Supports both positive and negated predicates (prefixed with 'not-', 'not_', '¬', or '!').
    """
    if env is None or not args:
        return False

    pred = predicate.lower().strip()
    negated = False
    for prefix in ("not-", "not_", "¬", "!"):
        if pred.startswith(prefix):
            negated = True
            pred = pred[len(prefix):].strip()
            break

    res = False
    if pred == "on":
        obj = args[0]
        target = args[1] if len(args) > 1 else "bin"
        if hasattr(env, "is_in_bin"):
            res = bool(env.is_in_bin(obj, target))

    elif pred == "holding":
        obj = args[0]
        if hasattr(env, "is_grasped"):
            res = bool(env.is_grasped(obj))

    elif pred in ("on-table", "on_table"):
        obj = args[0]
        is_held = False
        if hasattr(env, "is_grasped"):
            is_held = bool(env.is_grasped(obj))
        is_in_bin = False
        if hasattr(env, "is_in_bin"):
            is_in_bin = bool(env.is_in_bin(obj, "bin") or env.is_in_bin(obj, "pot"))
        res = (not is_held) and (not is_in_bin)

    return (not res) if negated else res


# ---------------------------------------------------------------------------
# Goal Check Node (Grounded in problem.pddl)
# ---------------------------------------------------------------------------
@register_condition("GoalCheck")
@register_condition("PDDLGoalCheck")
class PDDLGoalCheck(py_trees.behaviour.Behaviour):
    """
    Behavior Tree condition node that evaluates whether all goal predicates
    from problem.pddl are currently satisfied in the simulation.
    """

    def __init__(
        self,
        name: str = "GoalCheck",
        args: Optional[Dict[str, Any]] = None,
        env: Any = None,
        **kwargs
    ):
        super().__init__(name=name)
        self.args = dict(args) if args else {}
        self.env = env
        self.goal_predicates: List[Tuple[str, ...]] = []

    def setup(self, **kwargs):
        """Bind environment when setup is called by the BehaviorTree runner."""
        self.env = kwargs.get("env", self.env)

    def initialise(self):
        """Retrieve goal predicates from args or extract directly from problem.pddl."""
        if "goal_predicates" in self.args:
            self.goal_predicates = self.args["goal_predicates"]
        else:
            problem_source = self.args.get("problem_path", self.args.get("problem_pddl"))
            self.goal_predicates = extract_goals_from_problem(problem_source)

        self.logger.debug(f"[{self.name}] Goal predicates: {self.goal_predicates}")

    def update(self) -> py_trees.common.Status:
        if self.env is None:
            self.logger.error(f"[{self.name}] No environment bound.")
            return py_trees.common.Status.FAILURE

        if not self.goal_predicates:
            self.initialise()

        if not self.goal_predicates:
            self.logger.warning(f"[{self.name}] No goal predicates found in problem.pddl.")
            return py_trees.common.Status.SUCCESS

        all_satisfied = True
        for pred, *pred_args in self.goal_predicates:
            holds = evaluate_pddl_predicate(self.env, pred, list(pred_args))
            if not holds:
                all_satisfied = False
                break

        if all_satisfied:
            self.logger.info(f"[{self.name}] All PDDL goal conditions satisfied: {self.goal_predicates}")
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.FAILURE


# ---------------------------------------------------------------------------
# Individual Predicate / Visual Condition Node
# ---------------------------------------------------------------------------
@register_condition("VisualCheck")
@register_condition("PDDLCondition")
class PDDLCondition(py_trees.behaviour.Behaviour):
    """
    Checks an individual PDDL predicate or optional vision-language query.
    """

    def __init__(
        self,
        name: str = "VisualCheck",
        args: Optional[Dict[str, Any]] = None,
        env: Any = None,
        camera: Any = None,
        vlm_query_fn: Optional[Callable[[Any, str], str]] = None,
        negated: bool = False,
        **kwargs
    ):
        super().__init__(name=name)
        self.args = dict(args) if args else {}
        if negated:
            self.args["negated"] = True
        self.env = env
        self.camera = camera
        self.vlm_query_fn = vlm_query_fn

    def setup(self, **kwargs):
        self.env = kwargs.get("env", self.env)

    def update(self) -> py_trees.common.Status:
        # 1. If VLM query function and camera are explicitly provided, run visual check
        if self.vlm_query_fn is not None and self.camera is not None:
            situation = self.args.get("true_situation", self.args.get("what_to_check", ""))
            frame = self.camera.get_frame() if hasattr(self.camera, "get_frame") else None
            prompt = f"Answer only 'yes' or 'no'. Is the following true: {situation}"
            try:
                answer = self.vlm_query_fn(frame, prompt).strip().lower()
                is_yes = answer.startswith("y")
                if self.args.get("negated", False):
                    is_yes = not is_yes
                return py_trees.common.Status.SUCCESS if is_yes else py_trees.common.Status.FAILURE
            except Exception as exc:
                self.logger.warning(f"[{self.name}] VLM query error ({exc}), falling back to PDDL evaluation.")

        # 2. Evaluate predicate directly in the simulation environment
        predicate = self.args.get("predicate", self.args.get("pred", ""))
        pred_args = self.args.get("args", [])
        is_negated = bool(self.args.get("negated", False))
        if predicate and pred_args and self.env is not None:
            holds = evaluate_pddl_predicate(self.env, predicate, pred_args)
            if is_negated:
                holds = not holds
            return py_trees.common.Status.SUCCESS if holds else py_trees.common.Status.FAILURE

        return py_trees.common.Status.SUCCESS


# ---------------------------------------------------------------------------
# Aliases
# ---------------------------------------------------------------------------
GoalCheck = PDDLGoalCheck
VisualCheck = PDDLCondition