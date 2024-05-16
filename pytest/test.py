import rusty_graph
import pandas as pd
import json
def print_dict(d):
    # Convert the dictionary to a JSON-formatted string, indented for readability
    formatted_str = json.dumps(d, indent=4)
    print(formatted_str)
kg = rusty_graph.KnowledgeGraph()
indices = kg.add_nodes(
    data=[['Skole A','0']], # Data
    columns=['navn','unique_id'], # Columns
    node_title_field="navn",
    node_type="Skole", # node_type: String,
    unique_id_field="unique_id", # unique_id_field: String,
    conflict_handling="update" # conflict_handling: String, 
)
indices = kg.add_nodes(
    data=[['Klasse A','0','5', "29-12-1985 12:32:54"], ['Klasse B','1','6', "30-12-1985 12:32:54"], ['Klasse C','2','2', "29-12-1985 12:32:53"]], # Data
    columns=['navn','unique_id', 'elever', 'start'], # Columns
    node_title_field="navn",
    node_type="Klasse", # node_type: String,
    unique_id_field="unique_id", # unique_id_field: String,
    conflict_handling="update", # conflict_handling: String, 
    column_types={'elever':'Int', 'start':'DateTime %d-%m-%Y %H:%M:%S'}
)

nums = kg.add_relationships(
    data=[['0','0'],['1','0'], ['2','0']], # Data
    columns=['klasse_id','skole_id'], # Columns
    relationship_type="klasse_i", # relationship_name: String,
    source_type="Klasse", # left_node_type: String,
    source_id_field="klasse_id", # left_unique_id_field: String,
    target_type="Skole", # right_node_type: String,
    target_id_field="skole_id", # right_unique_id_field: String,
)
node_data = kg.get_nodes(node_type=None,filters=[{'title':'Klasse B'}])  # Replace "3144" with an actual unique_id from your data
print(node_data)
#print_dict(kg.get_node_attributes(node_data))
#print("ATTRIBUTES OF ",[node_data],":")
#print_dict(kg.get_node_attributes(node_data, ["title","elever","outgoing_relations"]))
school = kg.get_nodes(node_type=None,filters=[{'title':'Skole A'}])
classes = kg.traverse_incoming(indices=school,relationship_type="klasse_i",sort_attribute='start',max_relations=1)
print("CLASSES", classes)
#print(classes)
print_dict(kg.get_node_attributes(classes,["title","elever", "start"]))

#kg.save_graph_to_file("test_file")