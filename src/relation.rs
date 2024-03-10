// In relation.rs

use std::collections::HashMap;

pub struct Relation {
    pub relation_type: String,
    pub attributes: HashMap<String, String>,  // Adding a field for attributes
}

impl Relation {
    pub fn new(name: &str, attributes: HashMap<String, String>) -> Self {
        Relation {
            relation_type: name.to_string(),
            attributes,
        }
    }
}
