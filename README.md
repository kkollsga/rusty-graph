# Rusty Graph
Rusty Graph is a Rust-based project that aims to empower the generation of high-performance knowledge graphs within Python environments. Specifically designed for aggregating and merging data from SQL databases, Rusty Graph facilitates the seamless transition of relational database information into structured knowledge graphs. By leveraging Rust's efficiency and Python's flexibility, Rusty Graph offers an optimal solution for data scientists and developers looking to harness the power of knowledge graphs in their data-driven applications.

## Key Features
- **Efficient Data Integration:** Easily import and merge data from SQL databases to construct knowledge graphs, optimizing for performance and scalability.
- **High-Performance Operations:** Utilize Rust's performance capabilities to handle graph operations, making Rusty Graph ideal for working with large-scale data.
- **Python Compatibility:** Directly integrate Rusty Graph into Python projects, allowing for a smooth workflow within Python-based data analysis and machine learning pipelines.
- **Flexible Graph Manipulation:** Create, modify, and query knowledge graphs with a rich set of features, catering to complex data structures and relationships.

## Installation
To integrate Rusty Graph into your Python project, ensure you have Rust and Cargo installed on your system. Follow these steps:

1. **Install Rust:**
Visit the official Rust website for instructions on installing Rust and Cargo.

2. **Clone Rusty Graph:**
Clone the Rusty Graph repository into your desired project directory:
```sh
git clone https://github.com/kkollsga/rusty_graph.git
```

3. **Build with Maturin:**
Within the rusty_graph directory, use maturin to build the project and make it accessible from Python:

```sh
cd rusty_graph
maturin develop
```

## Usage
Start using Rusty Graph in your Python code to build and manipulate knowledge graphs:

```python
import rusty_graph

# Initialize a new KnowledgeGraph instance
kg = rusty_graph.KnowledgeGraph()

# Add nodes and relationships from SQL data
kg.add_node_from_sql("your_sql_query_for_nodes")
kg.add_relationship_from_sql("your_sql_query_for_relationships")

# Query the knowledge graph
node_info = kg.get_node_by_id("node_id")
```

## Contributing
We welcome contributions to Rusty Graph! If you have suggestions, bug reports, or would like to contribute code, please open an issue or a pull request on our GitHub repository.

## License
Rusty Graph is released under the MIT License. You are free to use, modify, and distribute it under the terms of this license.

