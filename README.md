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

### Getting Started
Start using Rusty Graph in your Python code to build and manipulate knowledge graphs:

```python
import rusty_graph
import pandas as pd

nodes_data = pd.DataFrame({
    "unique_id": ["1", "2"],
    "name": ["Node A", "Node B"],
    "design": ["Type X", "Type Y"]
})
# Initialize a new KnowledgeGraph instance
kg = rusty_graph.KnowledgeGraph()

# Add nodes and relationships from SQL data
kg.add_nodes(
    data=nodes_data.astype(str).to_numpy().tolist(),  # Convert DataFrame to list of lists
    columns=list(nodes_data.columns),  # Use DataFrame column names
    node_type="MyNodeType",
    unique_id_field="unique_id",
    conflict_handling="update"  # Options: "update", "replace", "skip"
)

# Query the knowledge graph
node_info = kg.get_node_by_id("1")
print(node_info)
```

### Advanced Usage with pandas and SQL
Rusty Graph allows for the efficient transformation of SQL database data into a knowledge graph. By leveraging pandas' SQL functionality, you can query your SQL database directly and use the resulting DataFrame with Rusty Graph. Here's an example that demonstrates this process:

```python
import rusty_graph
import pandas as pd
from sqlalchemy import create_engine

# Create a database engine
engine = create_engine('sqlite:///your_database.db')  # Adjust for your database

# Execute SQL query and load data into a pandas DataFrame
df = pd.read_sql_query("SELECT unique_id, attribute1, attribute2 FROM your_table", engine)

# Initialize Rusty Graph
kg = rusty_graph.KnowledgeGraph()

# Add nodes to the knowledge graph from DataFrame
kg.add_nodes(
    df.astype(str).to_numpy().tolist(),  # Data from DataFrame as list of lists
    list(df.columns),  # Column names
    "NodeType",  # Type of nodes
    "unique_id",  # Field for unique identifier
    "update"  # Conflict handling: "update", "replace", or "skip"
)

# Execute another SQL query for relationship data
relationship_df = pd.read_sql_query("SELECT source_id, target_id, relation_attribute FROM relationships_table", engine)

# Add relationships to the knowledge graph
kg.add_relationships(
    relationship_df.astype(str).to_numpy().tolist(),  # Data from DataFrame
    list(relationship_df.columns),  # Column names
    "RELATIONSHIP_TYPE",  # Name for the relationship type
    "SourceNodeType",  # Source node type
    "source_id",  # Source node unique identifier
    "TargetNodeType",  # Target node type
    "target_id",  # Target node unique identifier
    "update"  # Conflict handling: "update", "replace", or "skip"
)

# Retrieve node data by unique identifier
node_data = kg.get_nodes_by_id("specific_unique_id")
print(node_data)
```

## Contributing
We welcome contributions to Rusty Graph! If you have suggestions, bug reports, or would like to contribute code, please open an issue or a pull request on our GitHub repository.

## License
Rusty Graph is released under the MIT License. You are free to use, modify, and distribute it under the terms of this license.

