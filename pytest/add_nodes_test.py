from rusty_graph import KnowledgeGraph, DataType
import pandas as pd

df = pd.read_csv("pytest/data/wellbore_exploration_all.csv")
nodes_data = df[[
    'wlbNpdidWellbore',
    'wlbWellboreName', 
    'wlbContent', 
    'wlbCompletionDate',
    'wlbFormationWithHc1',
    'wlbAgeWithHc1'
]]
kg = KnowledgeGraph()
indices = kg.add_nodes(
    data=nodes_data.T.astype(str).to_numpy().tolist(), # Data
    columns=list(nodes_data.columns), # Columns
    node_title_field="wlbWellboreName",
    node_type="Well", # node_type: String,
    unique_id_field="wlbNpdidWellbore", # unique_id_field: String,
    conflict_handling="update", # conflict_handling: String, 
    column_types={'wlbCompletionDate':DataType.Date}
)

node_data = kg.get_nodes('wlbAgeWithHc1', 'PALEOCENE')  # Replace "3144" with an actual unique_id from your data
print(node_data)

for index in node_data:
        node_attributes = kg.get_node_attributes(indices=[index], specified_attributes=None)
        print(f"Node {index} Attributes: {node_attributes}")
        break