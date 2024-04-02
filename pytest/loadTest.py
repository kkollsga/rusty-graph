import rusty_graph
import pandas as pd
import json
def print_dict(d):
    # Convert the dictionary to a JSON-formatted string, indented for readability
    formatted_str = json.dumps(d, indent=4)
    print(formatted_str)
kg = rusty_graph.KnowledgeGraph()
kg.load_graph_from_file("test_file")

school = kg.get_nodes('title', 'Skole A')
classes = kg.traverse_incoming(indices=school,relationship_type="klasse_i",sort_attribute='start',ascending=False)
print("CLASSES")
#print(classes)
print_dict(kg.get_node_attributes(classes,["title","elever", "start"]))