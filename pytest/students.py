import pandas as pd
import rusty_graph

classes = pd.DataFrame(columns=["class_id","title","school_year"], data=[[0,"Class A","Year 7"],[1, "Class B", "Year 8"]])
students = pd.DataFrame(columns=["student_id", "name", "age", "class_id"], data=[[1,"Zack",12,0],[2,"Kevin",13,0],[3,"Alice",13,0],[4,"Xavier",12,0],[5,"Victor",14,1],[6,"Penny",13,1],[7,"Samuel",13],[8,"Luna",14,1],[9,"Hannah",14,1],[10,"Nina",14,1],[11,"Charlie",14,1]])
subjects = pd.DataFrame(columns=["subject_id", "subject_name"], data=[[1, "Maths"],[2, "Chemistry"]])
subject_records = pd.DataFrame(columns=["subject_record_id", "title", "student_id","subject_id","score","attendance"], data=[[
    0,"Zack_Maths",1,1,53.8,0.79],[1,"Zack_Chemistry",1,2,86.0,0.74],[2,"Kevin_Maths",2,1,70.1,0.83],[3,"Kevin_Chemistry",2,2,84.1,0.74],[4,"Alice_Maths",3,1,96.0,0.79],[5,"Alice_Chemistry",3,2,74.9,0.81],[6,"Xavier_Maths",4,1,52.9,0.98],
    [7,"Xavier_Chemistry",4,2,44.2,0.81],[8,"Victor_Maths",5,1,42.2,0.81],[9,"Victor_Chemistry",5,2,51.4,0.77],[10,"Penny_Maths",6,1,39.8,0.73],[11,"Penny_Chemistry",6,2,39.5,0.84],[12,"Samuel_Maths",7,1,43.9,0.90],[13,"Samuel_Chemistry",7,2,59.1,0.94],
    [14,"Luna_Maths",8,1,78.0,0.73],[15,"Luna_Chemistry",8,2,53.3,0.92],[16,"Hannah_Maths",9,1,78.6,0.87],[17,"Hannah_Chemistry",9,2,89.1,0.77],[18,"Nina_Maths",10,1,64.7,0.99],[19,"Nina_Chemistry",10,2,54.8,0.74],[20,"Charlie_Maths",11,1,86.9,0.88],
    [21,"Charlie_Chemistry",11,2,91.0,0.98]])

kg = rusty_graph.KnowledgeGraph()
kg.add_nodes(classes,"Class",'class_id','title')
kg.add_nodes(students,"Student","student_id","name")
kg.add_nodes(subjects,"Subject","subject_id","subject_name")
kg.add_nodes(subject_records,"Subject Record","subject_record_id","title")
kg.add_relationships(students, "ENROLLED_IN", "Student", "student_id", "Class", "class_id")
kg.add_relationships(subject_records, "HAS_SUBJECT", "Student", "student_id", "Subject","subject_id")
kg.add_relationships(subject_records, "HAS_RECORD", "Subject", "subject_id", "Subject Record","subject_record_id")
kg.add_relationships(subject_records, "EVALUATED_STUDENT", "Subject Record","subject_record_id", "Student", "student_id")
print("Unsorted: ", kg.type_filter("Student").get_relationships())