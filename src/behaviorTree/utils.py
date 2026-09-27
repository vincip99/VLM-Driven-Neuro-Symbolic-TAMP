"""
Behavior Tree Visualization and Presentation Utilities.
Provides styling and export helpers tailored for LaTeX-Beamer presentations.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from typing import Dict, Optional
import py_trees


def generate_presentation_dot(root: py_trees.behaviour.Behaviour) -> str:
    """
    Generate a high-clarity presentation-oriented Graphviz DOT string
    tailored for LaTeX-Beamer slides and papers (clean Helvetica font,
    curated pastel colors, multi-line labels, and optimal layout).
    """
    lines = [
        'digraph pastafarianism {',
        'ordering=out;',
        'graph [fontname="Helvetica", ranksep=0.38, nodesep=0.14];',
        'node [fontname="Helvetica", fontsize=12.5, fontname="Helvetica-Bold", fontcolor=black, penwidth=1.4];',
        'edge [fontname="Helvetica", penwidth=1.4, arrowsize=0.75];',
        '',
    ]

    def get_id(node):
        return f"node_{str(node.id).replace('-', '_')}"

    def format_node(node):
        name = node.name
        nid = get_id(node)

        if "goal_guarded_root" in name:
            lbl = "? goal_guarded_root\\n(Selector)"
            attr = 'shape=box, style="filled,rounded", fillcolor="#b2ebf2", fontcolor=black'
        elif "GoalCheck_Pre" in name:
            lbl = "GoalCheck_Pre\\n[Pre-Condition]"
            attr = 'shape=ellipse, style=filled, fillcolor="#c8e6c9", fontcolor="#1b5e20"'
        elif "GoalCheck_Post" in name:
            lbl = "GoalCheck_Post\\n[Post-Condition]"
            attr = 'shape=ellipse, style=filled, fillcolor="#c8e6c9", fontcolor="#1b5e20"'
        elif "retry_until_goal" in name:
            lbl = "retry_until_goal\\n[Retry <= 10]"
            attr = 'shape=box, style="filled,rounded", fillcolor="#e1bee7", fontcolor="#4a148c"'
        elif name == "attempt":
            lbl = "-> attempt\\n(Sequence)"
            attr = 'shape=box, style="filled,rounded", fillcolor="#ffe0b2", fontcolor=black'
        elif name == "pddl_plan_sequence":
            lbl = "-> pddl_plan_sequence\\n(Sequence Memory)"
            attr = 'shape=box, style="filled,rounded", fillcolor="#ffcc80", fontcolor=black'
        elif name.startswith("PickSeq("):
            arg = name[8:-1]
            lbl = f"-> PickSeq\\n({arg})"
            attr = 'shape=box, style="filled,rounded", fillcolor="#ffe0b2", fontcolor=black'
        elif name.startswith("PlaceSeq("):
            arg = name[9:-1].replace("->", " -> ")
            lbl = f"-> PlaceSeq\\n({arg})"
            attr = 'shape=box, style="filled,rounded", fillcolor="#ffe0b2", fontcolor=black'
        elif name.startswith("OnTable("):
            arg = name[8:-1]
            lbl = f"OnTable\\n({arg})"
            attr = 'shape=ellipse, style=filled, fillcolor="#c8e6c9", fontcolor="#1b5e20"'
        elif name.startswith("¬OnTable(") or name.startswith("NotOnTable("):
            arg = name.split("(", 1)[1][:-1]
            lbl = f"¬OnTable\\n({arg})"
            attr = 'shape=ellipse, style=filled, fillcolor="#c8e6c9", fontcolor="#1b5e20"'
        elif name.startswith("Holding("):
            arg = name[8:-1]
            lbl = f"Holding\\n({arg})"
            attr = 'shape=ellipse, style=filled, fillcolor="#c8e6c9", fontcolor="#1b5e20"'
        elif name.startswith("¬Holding(") or name.startswith("NotHolding("):
            arg = name.split("(", 1)[1][:-1]
            lbl = f"¬Holding\\n({arg})"
            attr = 'shape=ellipse, style=filled, fillcolor="#c8e6c9", fontcolor="#1b5e20"'
        elif name.startswith("On("):
            arg = name[3:-1].replace("->", " -> ")
            lbl = f"On\\n({arg})"
            attr = 'shape=ellipse, style=filled, fillcolor="#c8e6c9", fontcolor="#1b5e20"'
        elif name.startswith("PickUp("):
            arg = name[7:-1]
            lbl = f"PickUp\\n({arg})"
            attr = 'shape=ellipse, style=filled, fillcolor="#eeeeee", fontcolor=black'
        elif name.startswith("PlaceInBin("):
            arg = name[11:-1].replace("->", " -> ")
            lbl = f"PlaceInBin\\n({arg})"
            attr = 'shape=ellipse, style=filled, fillcolor="#eeeeee", fontcolor=black'
        elif isinstance(node, py_trees.composites.Selector):
            lbl = f"? {name}\\n(Selector)"
            attr = 'shape=box, style="filled,rounded", fillcolor="#b2ebf2", fontcolor=black'
        elif isinstance(node, py_trees.composites.Sequence):
            lbl = f"-> {name}\\n(Sequence)"
            attr = 'shape=box, style="filled,rounded", fillcolor="#ffe0b2", fontcolor=black'
        elif isinstance(node, py_trees.decorators.Decorator):
            lbl = f"{name}\\n[Decorator]"
            attr = 'shape=box, style="filled,rounded", fillcolor="#e1bee7", fontcolor="#4a148c"'
        else:
            lbl = name
            attr = 'shape=box, style="filled,rounded", fillcolor="#ffffff"'

        return f'{nid} [label="{lbl}", {attr}];'

    node_defs = []
    edges = []

    def traverse(node):
        node_defs.append(format_node(node))
        nid = get_id(node)
        for child in getattr(node, "children", []):
            cid = get_id(child)
            edges.append(f"{nid} -> {cid};")
            traverse(child)

    traverse(root)
    lines.extend(node_defs)
    lines.append("")
    lines.extend(edges)
    lines.append("}")
    return "\n".join(lines)


def render_presentation_bt(
    root: py_trees.behaviour.Behaviour,
    name: str = "behavior_tree_main_scenario",
    target_dir: str = os.path.join("Docs", "pictures"),
    sync_main_scenario_bt: bool = True,
) -> Dict[str, str]:
    """
    Render a presentation-grade Behavior Tree diagram for LaTeX-Beamer slides.
    Generates .dot, .pdf (vector), .png (300 DPI), and .svg files.
    Applies leaf staggering to preserve readable aspect ratios for presentation slides.

    Args:
        root: The py_trees Behaviour Tree root node.
        name: Base file name (default: "behavior_tree_main_scenario").
        target_dir: Target output directory.
        sync_main_scenario_bt: Also mirror to "main_scenario_bt.*" for Docs/presentation.tex.

    Returns:
        Dict mapping format extension ("dot", "pdf", "png", "svg") to absolute paths.
    """
    os.makedirs(target_dir, exist_ok=True)
    dot_content = generate_presentation_dot(root)

    dot_path = os.path.abspath(os.path.join(target_dir, f"{name}.dot"))
    pdf_path = os.path.abspath(os.path.join(target_dir, f"{name}.pdf"))
    png_path = os.path.abspath(os.path.join(target_dir, f"{name}.png"))
    svg_path = os.path.abspath(os.path.join(target_dir, f"{name}.svg"))

    # Preprocess with unflatten to prevent excessive horizontal spread in presentation slides
    render_dot_bytes = dot_content.encode("utf-8")
    try:
        res = subprocess.run(["unflatten", "-l", "3"], input=render_dot_bytes, capture_output=True, check=True)
        if res.stdout:
            render_dot_bytes = res.stdout
    except Exception as exc:
        print(f"[BehaviorTree] Note: unflatten pre-processor not applied ({exc})")

    with open(dot_path, "wb") as f:
        f.write(render_dot_bytes)

    try:
        subprocess.run(["dot", "-Tpdf", "-o", pdf_path], input=render_dot_bytes, check=True)
        subprocess.run(["dot", "-Tpng", "-Gdpi=300", "-o", png_path], input=render_dot_bytes, check=True)
        subprocess.run(["dot", "-Tsvg", "-o", svg_path], input=render_dot_bytes, check=True)
        print(f"[BehaviorTree] Saved presentation diagram to: {pdf_path} and {png_path}")

        # Keep main_scenario_bt in sync for Docs/presentation.tex
        if sync_main_scenario_bt and name != "main_scenario_bt":
            for ext in ["dot", "pdf", "png", "svg"]:
                src = os.path.join(target_dir, f"{name}.{ext}")
                dst = os.path.join(target_dir, f"main_scenario_bt.{ext}")
                if os.path.exists(src):
                    shutil.copy2(src, dst)

    except Exception as e:
        print(f"[BehaviorTree] Error rendering presentation diagram: {e}")

    return {"dot": dot_path, "pdf": pdf_path, "png": png_path, "svg": svg_path}


def export_presentation_bt(
    plan: Optional[Union[str, List[str]]] = None,
    root: Optional[py_trees.behaviour.Behaviour] = None,
    name: str = "behavior_tree_main_scenario",
    target_dir: str = os.path.join("Docs", "pictures"),
    env: Any = None,
    wrap_goal_check: bool = True,
    sync_main_scenario_bt: bool = True,
) -> Dict[str, str]:
    """
    Build (if root not provided) and export a presentation-grade Behavior Tree diagram
    in DOT, PDF (vector), PNG (300 DPI), and SVG formats tailored for LaTeX-Beamer.

    Args:
        plan: Optional PDDL plan string or list of action strings. If None and root is None,
              uses the standard main scenario benchmark plan.
        root: Optional pre-built Behavior Tree root.
        name: Base file name (default: "behavior_tree_main_scenario").
        target_dir: Directory to save diagrams (default: "Docs/pictures").
        env: Optional environment instance.
        wrap_goal_check: Whether to wrap plan in reactive goal-check / retry loop.
        sync_main_scenario_bt: Also mirror to "main_scenario_bt.*" for Docs/presentation.tex.

    Returns:
        Dict mapping format extension ("dot", "pdf", "png", "svg") to absolute paths.
    """
    if root is None:
        from .builder import build_bt_from_pddl_plan

        if plan is None:
            plan = [
                "(pick yellow_cube)",
                "(place yellow_cube pot)",
                "(pick red_can)",
                "(place red_can sorting_bin)",
                "(pick blue_can)",
                "(place blue_can sorting_bin)",
            ]

        root = build_bt_from_pddl_plan(
            plan=plan,
            env=env,
            wrap_goal_check=wrap_goal_check,
            render=False,
        )

    return render_presentation_bt(
        root=root,
        name=name,
        target_dir=target_dir,
        sync_main_scenario_bt=sync_main_scenario_bt,
    )
