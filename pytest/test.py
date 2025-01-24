import rusty_graph
import pandas as pd
print("rusty_graph: ", rusty_graph.__version__)
school = pd.DataFrame(columns=['id','title','age','students', 'score'], data=[[0,'Class A', 12, 23, 'A'], [1,'Class B', 13, 21, 'B'], [2,'Class C', 14, 30, 'D'], [3,'Class D', 14, 30, 'C']])

kg = rusty_graph.KnowledgeGraph()
nums = kg.add_nodes(
    data=school,
    node_type="Class",  # Type of node (example: Artist)
    unique_id_field='id', # Column name of unique identifier
    node_title_field='title',
    conflict_handling="update"  # Conflict handling: "update", "replace", or "skip"
)
print(nums)
print(kg.get_attributes())
print("Unsorted: ", kg.get_title())
print("Sorted: ", kg.sort('students').get_title())
print("Reverse Sorted: ", kg.sort('students', False).get_title())
print("Sorted: ", kg.sort_by(['students', 'score']).get_title())
print("Sorted: ", kg.sort_by(['students', ['score', False]]).get_title(3))