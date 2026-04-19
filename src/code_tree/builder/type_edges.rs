//! IMPLEMENTS / EXTENDS / HAS_METHOD edges from TypeRelationships.
//!
//! Also produces synthetic external Trait / Class node records for
//! target types referenced but not defined locally.

use crate::code_tree::models::TypeRelationship;
use std::collections::{HashMap, HashSet};

pub struct ImplementsEdge {
    pub type_name: String,
    pub interface_name: String,
}

pub struct ExtendsEdge {
    pub child_name: String,
    pub parent_name: String,
}

pub struct HasMethodEdge {
    pub owner: String,
    pub method: String,
}

pub struct ExternalNode {
    pub qualified_name: String,
    pub name: String,
}

pub struct TypeEdgeOutput {
    pub implements: Vec<ImplementsEdge>,
    pub extends: Vec<ExtendsEdge>,
    pub has_method: Vec<HasMethodEdge>,
    pub external_traits: Vec<ExternalNode>,
    pub external_classes: Vec<ExternalNode>,
}

fn resolve<'a>(name: &'a str, name_to_qname: &'a HashMap<String, String>) -> &'a str {
    name_to_qname.get(name).map(|s| s.as_str()).unwrap_or(name)
}

fn resolve_owner(qualified_name: &str, name_to_qname: &HashMap<String, String>) -> String {
    for sep in ["::", "."] {
        if let Some(idx) = qualified_name.rfind(sep) {
            let owner = &qualified_name[..idx];
            let simple = owner
                .rfind(sep)
                .map(|i| &owner[i + sep.len()..])
                .unwrap_or(owner);
            return name_to_qname
                .get(simple)
                .cloned()
                .unwrap_or_else(|| owner.to_string());
        }
    }
    qualified_name.to_string()
}

/// Build IMPLEMENTS/EXTENDS/HAS_METHOD edges + external trait/class stubs.
pub fn build_type_edges(
    relationships: &[TypeRelationship],
    known_interfaces: &HashSet<String>,
    known_classes: &HashSet<String>,
    name_to_qname: &mut HashMap<String, String>,
) -> TypeEdgeOutput {
    let mut implements = Vec::new();
    let mut extends = Vec::new();
    let mut has_method = Vec::new();
    let mut seen_impl: HashSet<(String, String)> = HashSet::new();
    let mut seen_ext: HashSet<(String, String)> = HashSet::new();
    let mut external_traits: HashMap<String, ExternalNode> = HashMap::new();
    let mut external_classes: HashMap<String, ExternalNode> = HashMap::new();

    for tr in relationships {
        match tr.relationship.as_str() {
            "inherent" => {
                for method in &tr.methods {
                    has_method.push(HasMethodEdge {
                        owner: resolve_owner(&method.qualified_name, name_to_qname),
                        method: method.qualified_name.clone(),
                    });
                }
            }
            "implements" => {
                let Some(target) = tr.target_type.as_ref() else {
                    continue;
                };
                let key = (tr.source_type.clone(), target.clone());
                if seen_impl.insert(key) {
                    let resolved_target = resolve(target, name_to_qname).to_string();
                    implements.push(ImplementsEdge {
                        type_name: resolve(&tr.source_type, name_to_qname).to_string(),
                        interface_name: resolved_target.clone(),
                    });
                    if !known_interfaces.contains(target) {
                        external_traits
                            .entry(resolved_target.clone())
                            .or_insert(ExternalNode {
                                qualified_name: resolved_target.clone(),
                                name: target.clone(),
                            });
                        name_to_qname
                            .entry(target.clone())
                            .or_insert(resolved_target);
                    }
                }
                for method in &tr.methods {
                    has_method.push(HasMethodEdge {
                        owner: resolve_owner(&method.qualified_name, name_to_qname),
                        method: method.qualified_name.clone(),
                    });
                }
            }
            "extends" => {
                let Some(target) = tr.target_type.as_ref() else {
                    continue;
                };
                let key = (tr.source_type.clone(), target.clone());
                if seen_ext.insert(key) {
                    let resolved_source = resolve(&tr.source_type, name_to_qname).to_string();
                    let resolved_target = resolve(target, name_to_qname).to_string();
                    extends.push(ExtendsEdge {
                        child_name: resolved_source.clone(),
                        parent_name: resolved_target.clone(),
                    });
                    if !known_classes.contains(&tr.source_type) {
                        external_classes
                            .entry(resolved_source.clone())
                            .or_insert(ExternalNode {
                                qualified_name: resolved_source,
                                name: tr.source_type.clone(),
                            });
                    }
                    if !known_classes.contains(target) {
                        external_classes
                            .entry(resolved_target.clone())
                            .or_insert(ExternalNode {
                                qualified_name: resolved_target,
                                name: target.clone(),
                            });
                    }
                }
            }
            _ => {}
        }
    }

    TypeEdgeOutput {
        implements,
        extends,
        has_method,
        external_traits: external_traits.into_values().collect(),
        external_classes: external_classes.into_values().collect(),
    }
}
