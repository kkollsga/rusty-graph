"""Build the tiny_graph.kgl fixture used by the pre-release test suite.

Spec: ~50 nodes of each of 3 types (Person, Company, Article), a
handful of connections, one text column per type for embedder
scoring. Stays under 1 MB so it can be regenerated quickly per
test session.

This module is import-friendly: `build_tiny_graph(target_path)`
returns nothing but writes a .kgl. Tests call it from a
`session`-scoped fixture in conftest.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import kglite


def build_tiny_graph(target_path: Path) -> None:
    """Build and persist the tiny_graph.kgl at `target_path`."""
    g = kglite.KnowledgeGraph()

    # 50 Person nodes with name, age, city + a docstring-style 'bio'
    # text column for embedder testing.
    cities = ["Oslo", "Bergen", "Trondheim", "Stavanger", "Tromsø"]
    person_data = pd.DataFrame(
        {
            "id": [f"p{i:03d}" for i in range(50)],
            "title": [f"Person{i}" for i in range(50)],
            "name": [f"Person{i}" for i in range(50)],
            "age": [20 + (i % 50) for i in range(50)],
            "city": [cities[i % len(cities)] for i in range(50)],
            "bio": [
                (
                    "researcher in quantum computing"
                    if i % 3 == 0
                    else "chef and baking enthusiast"
                    if i % 3 == 1
                    else "data scientist focused on graph theory"
                )
                for i in range(50)
            ],
        }
    )
    g.add_nodes(person_data, "Person", "id", node_title_field="name")

    company_data = pd.DataFrame(
        {
            "id": [f"c{i:03d}" for i in range(50)],
            "title": [f"Company{i}" for i in range(50)],
            "industry": [["tech", "finance", "energy", "biotech"][i % 4] for i in range(50)],
            "description": [
                "develops machine learning infrastructure"
                if i % 2 == 0
                else "operates oil and gas exploration platforms"
                for i in range(50)
            ],
        }
    )
    g.add_nodes(company_data, "Company", "id", node_title_field="title")

    # Three semantic clusters with within-cluster variation so the
    # embedder produces distinguishable scores even on the truncated
    # corpus. Tests assume the "quantum" cluster ranks highest for the
    # query "quantum mechanics".
    quantum_bodies = [
        "Introduction to quantum mechanics, superposition, and entanglement.",
        "Quantum gates and qubit measurement in a small quantum circuit.",
        "Schrödinger's equation explained for chemistry students.",
        "Quantum tunneling and barrier penetration in solid-state physics.",
        "Wave-particle duality and the double-slit experiment.",
        "Quantum decoherence and the measurement problem.",
        "Bell's inequality and tests of local realism.",
        "Path integral formulation of quantum field theory.",
        "Quantum information theory: qubits, gates, and algorithms.",
        "Shor's algorithm and integer factorisation on quantum computers.",
        "Grover's algorithm for unstructured database search.",
        "Quantum entanglement applied to cryptographic key distribution.",
        "Quantum error correction with stabiliser codes.",
        "Adiabatic quantum computing for optimisation problems.",
        "Topological quantum computing with anyon braiding.",
        "Variational quantum eigensolvers for chemistry simulation.",
        "Quantum supremacy benchmarks on superconducting qubits.",
    ]
    baking_bodies = [
        "How to start a sourdough starter from scratch.",
        "Hydration ratios for high-protein bread flour.",
        "Autolyse step and gluten development in artisan loaves.",
        "Proofing techniques: bulk fermentation and final proof.",
        "Steam injection in a home oven for crusty bread.",
        "Cold retard fermentation overnight for flavour development.",
        "Scoring patterns for visually striking bread loaves.",
        "Dutch oven vs baking stone: which gives a better crust.",
        "Whole wheat sourdough techniques and grain blends.",
        "Brioche dough enriched with butter and eggs.",
        "Croissant lamination: butter blocks and tri-folds.",
        "Focaccia dough with high hydration and olive oil.",
        "Pita bread that puffs properly in a hot oven.",
        "Pretzel boiling solution and lye safety considerations.",
        "Bagel boiling vs baking-soda alternatives.",
        "Naan bread on a stovetop in a cast-iron skillet.",
    ]
    code_bodies = [
        "Python decorator patterns for memoisation and caching.",
        "Rust ownership rules and lifetime annotations.",
        "Asynchronous programming with async/await in modern languages.",
        "Type systems: parametric polymorphism vs dependent types.",
        "Garbage collection algorithms: tracing vs reference counting.",
        "Compiler optimisations: inlining, dead-code elimination, LTO.",
        "Database transactions and the ACID properties.",
        "Distributed consensus algorithms: Paxos and Raft.",
        "Functional programming and immutable data structures.",
        "Memory models: sequential consistency vs weak ordering.",
        "Web frameworks: synchronous vs asynchronous request handling.",
        "Static analysis: type checkers, linters, and provers.",
        "Build systems and incremental compilation.",
        "Testing strategies: unit, integration, property-based.",
        "Containerisation: namespaces, cgroups, and overlay filesystems.",
        "Cryptographic primitives: hashing, signing, encryption.",
        "Networking protocols: TCP, QUIC, and HTTP/3.",
    ]
    bodies = quantum_bodies + baking_bodies + code_bodies
    article_data = pd.DataFrame(
        {
            "id": [f"a{i:03d}" for i in range(50)],
            "title": ["Quantum" if i < 17 else "Baking" if i < 33 else "Programming" for i in range(50)],
            "body": [bodies[i] for i in range(50)],
        }
    )
    g.add_nodes(article_data, "Article", "id", node_title_field="title")

    # A few KNOWS connections (Person → Person, sparse).
    knows = pd.DataFrame(
        {
            "from": [f"p{i:03d}" for i in range(0, 49, 7)],
            "to": [f"p{(i + 3) % 50:03d}" for i in range(0, 49, 7)],
        }
    )
    g.add_connections(knows, "KNOWS", "Person", "from", "Person", "to")

    # WORKS_AT (Person → Company)
    works = pd.DataFrame(
        {
            "from": [f"p{i:03d}" for i in range(0, 50, 5)],
            "to": [f"c{i % 10:03d}" for i in range(0, 50, 5)],
        }
    )
    g.add_connections(works, "WORKS_AT", "Person", "from", "Company", "to")

    # WROTE (Person → Article)
    wrote = pd.DataFrame(
        {
            "from": [f"p{i:03d}" for i in range(0, 50, 3)],
            "to": [f"a{i % 20:03d}" for i in range(0, 50, 3)],
        }
    )
    g.add_connections(wrote, "WROTE", "Person", "from", "Article", "to")

    g.save(str(target_path))


if __name__ == "__main__":
    import sys

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tiny_graph.kgl")
    build_tiny_graph(out)
    print(f"built {out} ({out.stat().st_size} bytes)")
