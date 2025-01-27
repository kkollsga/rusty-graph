import rusty_graph
import pandas as pd
print("rusty_graph: ", rusty_graph.__version__)
school = pd.DataFrame(
    columns=['id','title'], 
    data=[
        [0,'Class A'], 
        [1,'Class B'], 
        [2,'Class C'], 
        [3,'Class D']
])

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
print("Sorted by student #: ", kg.sort('students').get_title())
print("Reverse Sorted by student #: ", kg.sort('students', False).get_title())
print("Sorted by students then score: ", kg.sort_by(['students', 'score']).get_title())
print("Sorted by students then score in reverse: ", kg.sort_by(['students', ['score', False]]).get_title(3))

print("Sorted by title: ", kg.sort('title', False).get_title())
print("Sorted by title: ", kg.sort('title').get_title())
print("Sorted by age: ", kg.sort('age').get_title())
