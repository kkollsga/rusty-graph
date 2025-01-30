# Rusty Graph
Rusty Graph is a Rust-based project that aims to empower the generation of high-performance knowledge graphs within Python environments. Specifically designed for aggregating and merging data from SQL databases, Rusty Graph facilitates the seamless transition of relational database information into structured knowledge graphs. By leveraging Rust's efficiency and Python's flexibility, Rusty Graph offers an optimal solution for data scientists and developers looking to harness the power of knowledge graphs in their data-driven applications.

## Key Features
- **Efficient Data Integration:** Easily import and merge data from SQL databases to construct knowledge graphs, optimizing for performance and scalability.
- **High-Performance Operations:** Utilize Rust's performance capabilities to handle graph operations, making Rusty Graph ideal for working with large-scale data.
- **Python Compatibility:** Directly integrate Rusty Graph into Python projects, allowing for a smooth workflow within Python-based data analysis and machine learning pipelines.
- **Flexible Graph Manipulation:** Create, modify, and query knowledge graphs with a rich set of features, catering to complex data structures and relationships.

## Direct Download and Install
Users can download the .whl file directly from the repository and install it using pip. 
- *Note that the release is only compatible with Python 3.12 on win_amd64.*
- *The library is still in alpha, so the functionality is very limited.*
```sh
pip install https://github.com/kkollsga/rusty_graph/blob/main/wheels/rusty_graph-0.1.34-cp312-cp312-win_amd64.whl?raw=true
# or upgrade an already installed library
pip install --upgrade https://github.com/kkollsga/rusty_graph/blob/main/wheels/rusty_graph-0.1.34-cp312-cp312-win_amd64.whl?raw=true
```


## Clone and Install
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
    node_type="MyNodeType",  # Type of node (example: Artist)
    unique_id_field="unique_id", # Column name of unique identifier
    node_title_field="name", # node title
    conflict_handling="update"  # Conflict handling: "update", "replace", or "skip"
)

# Query the knowledge graph
matching_nodes = kg.get_nodes(node_type=None, filters=[{"unique_id": "1"}])
print(matching_nodes)
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
    data=df.astype(str).to_numpy().tolist(),  # Data from DataFrame as list of lists
    columns=list(df.columns),  # Column names
    node_type="NodeType",  # Type of node (example: Artist)
    unique_id_field="unique_id",  # Column name of unique identifier
    node_title_field="title", # node title
    conflict_handling="update"  # Conflict handling: "update", "replace", or "skip"
)

# Execute another SQL query for relationship data
relationship_df = pd.read_sql_query("SELECT source_id, target_id, relation_attribute FROM relationships_table", engine)

# Add relationships to the knowledge graph
kg.add_relationships(
    data=relationship_df.astype(str).to_numpy().tolist(),  # Data from DataFrame
    columns=list(relationship_df.columns),  # Column names
    relationship_type="RELATIONSHIP_TYPE",  # Name for the relationship type
    source_type="SourceNodeType",  # Source node type
    source_id_field="source_id",  # Column name of source node unique identifier
    source_title_field= "source_title", # Source title
    target_type="TargetNodeType",  # Target node type
    target_id_field="target_id",  # Column name of target node unique identifier
    target_title_field= "target_title", # Source title
    conflict_handling="update"  # Conflict handling: "update", "replace", or "skip"
)
kg.save_to_file("KG.bin")
# Retrieve node data by unique identifier
matching_nodes = kg.get_nodes(node_type=None, filters=[{"title": "specific_title_name"}])
print(matching_nodes)
```

### Traverse Graph and return node properties
```python
import rusty_graph
kg = rusty_graph.KnowledgeGraph()
# Get all relationships found in selected nodes
kg.load_from_file("KG.bin")
unique_relationships = kg.get_relationships(matching_nodes)
print(unique_relationships)

# Traverse graph, and return matching nodes
outgoing_nodes = kg.traverse_outgoing(matching_nodes, 'MADE_DISCOVERY')
incoming_nodes = kg.traverse_incoming(matching_nodes, 'DRILLED_BY')

# Get values
print(kg.get_node_attributes(outgoing_nodes, ['title']))
```

## Contributing
We welcome contributions to Rusty Graph! If you have suggestions, bug reports, or would like to contribute code, please open an issue or a pull request on our GitHub repository.

## License
Rusty Graph is released under the MIT License. You are free to use, modify, and distribute it under the terms of this license.

