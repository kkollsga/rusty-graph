#!/usr/bin/env python3
"""
Example: Building a knowledge graph with KGLite.

Demonstrates how to create nodes, connections, indexes, and run queries
using both the fluent API and Cypher. The example builds a small company
knowledge graph with employees, departments, projects, and skills.

Usage:
    python build_graph.py
"""

import kglite
import pandas as pd

# ── 1. Create a graph ────────────────────────────────────────────────────

graph = kglite.KnowledgeGraph()

# ── 2. Add nodes via DataFrame (bulk loading) ────────────────────────────

employees = pd.DataFrame([
    {"emp_id": "E001", "name": "Alice Chen",    "role": "Engineer",        "salary": 95000},
    {"emp_id": "E002", "name": "Bob Martinez",  "role": "Engineer",        "salary": 92000},
    {"emp_id": "E003", "name": "Carol Davis",   "role": "Designer",        "salary": 88000},
    {"emp_id": "E004", "name": "Dan Lee",       "role": "Product Manager", "salary": 105000},
    {"emp_id": "E005", "name": "Eve Johnson",   "role": "Engineer",        "salary": 98000},
    {"emp_id": "E006", "name": "Frank Wu",      "role": "Data Scientist",  "salary": 100000},
])

departments = pd.DataFrame([
    {"dept_id": "D01", "name": "Engineering", "budget": 500000},
    {"dept_id": "D02", "name": "Design",      "budget": 200000},
    {"dept_id": "D03", "name": "Product",     "budget": 300000},
])

projects = pd.DataFrame([
    {"project_id": "P01", "name": "Search Rewrite",   "status": "active",    "priority": 1},
    {"project_id": "P02", "name": "Mobile App",       "status": "active",    "priority": 2},
    {"project_id": "P03", "name": "Dashboard",        "status": "completed", "priority": 3},
])

skills = pd.DataFrame([
    {"skill_id": "S01", "name": "Python"},
    {"skill_id": "S02", "name": "Rust"},
    {"skill_id": "S03", "name": "JavaScript"},
    {"skill_id": "S04", "name": "SQL"},
    {"skill_id": "S05", "name": "Figma"},
])

# Load all node types
graph.add_nodes(data=employees,   node_type="Employee",   unique_id_field="emp_id",     node_title_field="name")
graph.add_nodes(data=departments, node_type="Department", unique_id_field="dept_id",    node_title_field="name")
graph.add_nodes(data=projects,    node_type="Project",    unique_id_field="project_id", node_title_field="name")
graph.add_nodes(data=skills,      node_type="Skill",      unique_id_field="skill_id",   node_title_field="name")

# ── 3. Add connections via DataFrame ─────────────────────────────────────

# Employee → Department (WORKS_IN)
works_in = pd.DataFrame([
    {"emp_id": "E001", "dept_id": "D01"},
    {"emp_id": "E002", "dept_id": "D01"},
    {"emp_id": "E003", "dept_id": "D02"},
    {"emp_id": "E004", "dept_id": "D03"},
    {"emp_id": "E005", "dept_id": "D01"},
    {"emp_id": "E006", "dept_id": "D01"},
])

graph.add_connections(
    data=works_in,
    connection_type="WORKS_IN",
    source_type="Employee",   source_id_field="emp_id",
    target_type="Department", target_id_field="dept_id",
)

# Employee → Project (WORKS_ON) with a "role" property on the edge
works_on = pd.DataFrame([
    {"emp_id": "E001", "project_id": "P01", "role": "lead"},
    {"emp_id": "E002", "project_id": "P01", "role": "contributor"},
    {"emp_id": "E002", "project_id": "P02", "role": "contributor"},
    {"emp_id": "E003", "project_id": "P02", "role": "lead"},
    {"emp_id": "E004", "project_id": "P01", "role": "stakeholder"},
    {"emp_id": "E004", "project_id": "P02", "role": "stakeholder"},
    {"emp_id": "E005", "project_id": "P03", "role": "lead"},
    {"emp_id": "E006", "project_id": "P01", "role": "contributor"},
])

graph.add_connections(
    data=works_on,
    connection_type="WORKS_ON",
    source_type="Employee", source_id_field="emp_id",
    target_type="Project",  target_id_field="project_id",
)

# Employee → Skill (HAS_SKILL)
has_skill = pd.DataFrame([
    {"emp_id": "E001", "skill_id": "S01"},
    {"emp_id": "E001", "skill_id": "S02"},
    {"emp_id": "E002", "skill_id": "S03"},
    {"emp_id": "E002", "skill_id": "S04"},
    {"emp_id": "E003", "skill_id": "S05"},
    {"emp_id": "E003", "skill_id": "S03"},
    {"emp_id": "E005", "skill_id": "S01"},
    {"emp_id": "E005", "skill_id": "S04"},
    {"emp_id": "E006", "skill_id": "S01"},
    {"emp_id": "E006", "skill_id": "S04"},
])

graph.add_connections(
    data=has_skill,
    connection_type="HAS_SKILL",
    source_type="Employee", source_id_field="emp_id",
    target_type="Skill",    target_id_field="skill_id",
)

# ── 4. Create nodes and connections via Cypher ───────────────────────────

# You can also build the graph with Cypher CREATE statements
graph.cypher("""
    CREATE (:Location {location_id: 'L01', name: 'San Francisco', country: 'US'})
""")
graph.cypher("""
    CREATE (:Location {location_id: 'L02', name: 'New York', country: 'US'})
""")

# Link departments to locations
graph.cypher("""
    MATCH (d:Department {dept_id: 'D01'}), (l:Location {location_id: 'L01'})
    CREATE (d)-[:LOCATED_IN]->(l)
""")
graph.cypher("""
    MATCH (d:Department {dept_id: 'D02'}), (l:Location {location_id: 'L02'})
    CREATE (d)-[:LOCATED_IN]->(l)
""")
graph.cypher("""
    MATCH (d:Department {dept_id: 'D03'}), (l:Location {location_id: 'L01'})
    CREATE (d)-[:LOCATED_IN]->(l)
""")

# ── 5. Schema overview ──────────────────────────────────────────────────

print("=== Schema ===")
schema = graph.schema()
print(f"Nodes: {schema['node_count']}, Edges: {schema['edge_count']}")
for nt in schema["node_types"]:
    print(f"  :{nt['label']} ({nt['count']} nodes)")
for ct in schema["connection_types"]:
    print(f"  -[:{ct['type']}]- ({ct['count']} edges)")

# ── 6. Querying with Cypher ──────────────────────────────────────────────

print("\n=== Engineers and their skills ===")
result = graph.cypher("""
    MATCH (e:Employee)-[:HAS_SKILL]->(s:Skill)
    WHERE e.role = 'Engineer'
    RETURN e.name AS engineer, collect(s.name) AS skills
    ORDER BY e.name
""")
for row in result:
    print(f"  {row['engineer']}: {', '.join(row['skills'])}")

print("\n=== Department headcount and avg salary ===")
result = graph.cypher("""
    MATCH (e:Employee)-[:WORKS_IN]->(d:Department)
    RETURN d.name AS department,
           count(e) AS headcount,
           avg(e.salary) AS avg_salary
    ORDER BY headcount DESC
""")
for row in result:
    print(f"  {row['department']}: {row['headcount']} people, avg ${row['avg_salary']:,.0f}")

print("\n=== Active projects with team members ===")
result = graph.cypher("""
    MATCH (e:Employee)-[:WORKS_ON]->(p:Project)
    WHERE p.status = 'active'
    RETURN p.name AS project, collect(e.name) AS team
    ORDER BY p.priority
""")
for row in result:
    print(f"  {row['project']}: {', '.join(row['team'])}")

print("\n=== Who shares a project with Alice? ===")
result = graph.cypher("""
    MATCH (alice:Employee {name: 'Alice Chen'})-[:WORKS_ON]->(p:Project)<-[:WORKS_ON]-(coworker:Employee)
    RETURN DISTINCT coworker.name AS name, collect(p.name) AS shared_projects
    ORDER BY name
""")
for row in result:
    print(f"  {row['name']} — shared: {', '.join(row['shared_projects'])}")

# ── 7. Graph algorithms ─────────────────────────────────────────────────

print("\n=== PageRank (top 5) ===")
pr = graph.pagerank(top_k=5)
for row in pr:
    print(f"  {row['title']} ({row['label']}): {row['score']:.4f}")

print("\n=== Shortest path: Alice → Dashboard project ===")
path = graph.shortest_path("Employee", "E001", "Project", "P03")
if path:
    titles = [n["title"] for n in path["path"]]
    print(f"  {' → '.join(titles)} (length: {path['length']})")

# ── 8. Fluent API (filter / traverse) ───────────────────────────────────

print("\n=== Python-skilled employees (fluent API) ===")
python_devs = (
    graph
    .type_filter("Skill")
    .filter({"name": "Python"})
    .traverse("HAS_SKILL", direction="incoming")
    .get_nodes()
)
for node in python_devs:
    print(f"  {node['name']} — {node['role']}")

# ── 9. Save the graph ───────────────────────────────────────────────────

output_path = "company_graph.kgl"
graph.save(output_path)
print(f"\nGraph saved to {output_path}")
print("Load it again with: graph = kglite.load('company_graph.kgl')")
