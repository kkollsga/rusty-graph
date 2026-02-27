# Common Recipes

Short, copy-paste examples for common tasks.

## Upsert with MERGE

```python
graph.cypher("""
    MERGE (p:Person {email: 'alice@example.com'})
    ON CREATE SET p.created = '2024-01-01', p.name = 'Alice'
    ON MATCH SET p.last_seen = '2024-01-15'
""")
```

## Top-K Nodes by Centrality

```python
top_nodes = graph.pagerank(top_k=10)
for node in top_nodes:
    print(f"{node['title']}: {node['score']:.3f}")
```

## 2-Hop Neighborhood

```python
graph.cypher("""
    MATCH (me:Person {name: 'Alice'})-[:KNOWS*2]-(fof:Person)
    WHERE fof <> me
    RETURN DISTINCT fof.name
""")
```

## Export Subgraph

```python
subgraph = (
    graph.select('Person')
    .where({'name': 'Alice'})
    .expand(hops=2)
    .to_subgraph()
)
subgraph.export('alice_network.graphml', format='graphml')
```

## Parameterized Queries

```python
graph.cypher(
    "MATCH (p:Person) WHERE p.city = $city AND p.age > $min_age RETURN p.name",
    params={'city': 'Oslo', 'min_age': 25}
)
```

## Delete Subgraph

```python
graph.cypher("""
    MATCH (u:User) WHERE u.status = 'inactive'
    DETACH DELETE u
""")
```

## Aggregation with Relationship Properties

```python
graph.cypher("""
    MATCH (p:Person)-[r:RATED]->(m:Movie)
    RETURN p.name, avg(r.score) AS avg_rating, count(m) AS movies_rated
    ORDER BY avg_rating DESC
""")
```
