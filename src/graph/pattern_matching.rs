// Pattern Matching Module for Cypher-like queries
// Supports patterns like: (p:Play)-[:HAS_PROSPECT]->(pr:Prospect)-[:BECAME_DISCOVERY]->(d:Discovery)

use std::collections::HashMap;
use petgraph::graph::NodeIndex;
use petgraph::Direction;
use petgraph::visit::EdgeRef;
use crate::datatypes::values::Value;
use crate::graph::schema::DirGraph;

// ============================================================================
// AST Types
// ============================================================================

/// A complete pattern to match against the graph
#[derive(Debug, Clone)]
pub struct Pattern {
    pub elements: Vec<PatternElement>,
}

/// Either a node or edge pattern
#[derive(Debug, Clone)]
pub enum PatternElement {
    Node(NodePattern),
    Edge(EdgePattern),
}

/// Pattern for matching nodes: (var:Type {prop: value})
#[derive(Debug, Clone)]
pub struct NodePattern {
    pub variable: Option<String>,
    pub node_type: Option<String>,
    pub properties: Option<HashMap<String, PropertyMatcher>>,
}

/// Pattern for matching edges: -[:TYPE {prop: value}]->
#[derive(Debug, Clone)]
pub struct EdgePattern {
    pub variable: Option<String>,
    pub connection_type: Option<String>,
    pub direction: EdgeDirection,
    pub properties: Option<HashMap<String, PropertyMatcher>>,
}

/// Direction of edge traversal
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EdgeDirection {
    Outgoing,  // -[]->
    Incoming,  // <-[]-
    Both,      // -[]-
}

/// Property value matcher (currently only equality)
#[derive(Debug, Clone)]
pub enum PropertyMatcher {
    Equals(Value),
}

// ============================================================================
// Match Results
// ============================================================================

/// A single pattern match with variable bindings
#[derive(Debug, Clone)]
pub struct PatternMatch {
    pub bindings: HashMap<String, MatchBinding>,
}

/// A bound value (either node or edge)
#[derive(Debug, Clone)]
pub enum MatchBinding {
    Node {
        #[allow(dead_code)]
        index: NodeIndex,
        node_type: String,
        title: String,
        id: Value,
        properties: HashMap<String, Value>,
    },
    Edge {
        source: NodeIndex,
        target: NodeIndex,
        connection_type: String,
        properties: HashMap<String, Value>,
    },
}

// ============================================================================
// Tokenizer
// ============================================================================

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    LParen,      // (
    RParen,      // )
    LBracket,    // [
    RBracket,    // ]
    LBrace,      // {
    RBrace,      // }
    Colon,       // :
    Comma,       // ,
    Dash,        // -
    GreaterThan, // >
    LessThan,    // <
    Identifier(String),
    StringLit(String),
    IntLit(i64),
    FloatLit(f64),
    BoolLit(bool),
}

pub fn tokenize(input: &str) -> Result<Vec<Token>, String> {
    let mut tokens = Vec::new();
    let mut chars = input.chars().peekable();

    while let Some(&ch) = chars.peek() {
        match ch {
            ' ' | '\t' | '\n' | '\r' => {
                chars.next();
            }
            '(' => {
                tokens.push(Token::LParen);
                chars.next();
            }
            ')' => {
                tokens.push(Token::RParen);
                chars.next();
            }
            '[' => {
                tokens.push(Token::LBracket);
                chars.next();
            }
            ']' => {
                tokens.push(Token::RBracket);
                chars.next();
            }
            '{' => {
                tokens.push(Token::LBrace);
                chars.next();
            }
            '}' => {
                tokens.push(Token::RBrace);
                chars.next();
            }
            ':' => {
                tokens.push(Token::Colon);
                chars.next();
            }
            ',' => {
                tokens.push(Token::Comma);
                chars.next();
            }
            '-' => {
                tokens.push(Token::Dash);
                chars.next();
            }
            '>' => {
                tokens.push(Token::GreaterThan);
                chars.next();
            }
            '<' => {
                tokens.push(Token::LessThan);
                chars.next();
            }
            '"' | '\'' => {
                let quote = ch;
                chars.next(); // consume opening quote
                let mut s = String::new();
                while let Some(&c) = chars.peek() {
                    if c == quote {
                        chars.next(); // consume closing quote
                        break;
                    }
                    if c == '\\' {
                        chars.next();
                        if let Some(&escaped) = chars.peek() {
                            s.push(match escaped {
                                'n' => '\n',
                                't' => '\t',
                                'r' => '\r',
                                _ => escaped,
                            });
                            chars.next();
                        }
                    } else {
                        s.push(c);
                        chars.next();
                    }
                }
                tokens.push(Token::StringLit(s));
            }
            c if c.is_ascii_digit() || c == '.' => {
                let mut num_str = String::new();
                let mut has_dot = false;
                while let Some(&c) = chars.peek() {
                    if c.is_ascii_digit() {
                        num_str.push(c);
                        chars.next();
                    } else if c == '.' && !has_dot {
                        has_dot = true;
                        num_str.push(c);
                        chars.next();
                    } else {
                        break;
                    }
                }
                if has_dot {
                    tokens.push(Token::FloatLit(
                        num_str.parse().map_err(|_| format!("Invalid float: {}", num_str))?
                    ));
                } else {
                    tokens.push(Token::IntLit(
                        num_str.parse().map_err(|_| format!("Invalid integer: {}", num_str))?
                    ));
                }
            }
            c if c.is_ascii_alphabetic() || c == '_' => {
                let mut ident = String::new();
                while let Some(&c) = chars.peek() {
                    if c.is_ascii_alphanumeric() || c == '_' {
                        ident.push(c);
                        chars.next();
                    } else {
                        break;
                    }
                }
                // Check for boolean literals
                match ident.to_lowercase().as_str() {
                    "true" => tokens.push(Token::BoolLit(true)),
                    "false" => tokens.push(Token::BoolLit(false)),
                    _ => tokens.push(Token::Identifier(ident)),
                }
            }
            _ => return Err(format!("Unexpected character: '{}'", ch)),
        }
    }

    Ok(tokens)
}

// ============================================================================
// Parser
// ============================================================================

pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Parser { tokens, pos: 0 }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn advance(&mut self) -> Option<&Token> {
        let token = self.tokens.get(self.pos);
        self.pos += 1;
        token
    }

    fn expect(&mut self, expected: &Token) -> Result<(), String> {
        match self.advance() {
            Some(token) if token == expected => Ok(()),
            Some(token) => Err(format!("Expected {:?}, got {:?}", expected, token)),
            None => Err(format!("Expected {:?}, got end of input", expected)),
        }
    }

    /// Parse a complete pattern: node (edge node)*
    pub fn parse_pattern(&mut self) -> Result<Pattern, String> {
        let mut elements = Vec::new();

        // Must start with a node pattern
        elements.push(PatternElement::Node(self.parse_node_pattern()?));

        // Parse edge-node pairs
        while self.peek().is_some() {
            // Check for edge pattern (starts with - or <)
            match self.peek() {
                Some(Token::Dash) | Some(Token::LessThan) => {
                    elements.push(PatternElement::Edge(self.parse_edge_pattern()?));
                    elements.push(PatternElement::Node(self.parse_node_pattern()?));
                }
                _ => break,
            }
        }

        Ok(Pattern { elements })
    }

    /// Parse node pattern: (var:Type {props})
    fn parse_node_pattern(&mut self) -> Result<NodePattern, String> {
        self.expect(&Token::LParen)?;

        let mut variable = None;
        let mut node_type = None;
        let mut properties = None;

        // Check what comes next
        match self.peek() {
            Some(Token::RParen) => {
                // Empty node pattern: ()
            }
            Some(Token::Colon) => {
                // No variable, just type: (:Type)
                self.advance(); // consume :
                if let Some(Token::Identifier(name)) = self.advance().cloned() {
                    node_type = Some(name);
                } else {
                    return Err("Expected type name after ':'".to_string());
                }
            }
            Some(Token::Identifier(_)) => {
                // Variable name
                if let Some(Token::Identifier(name)) = self.advance().cloned() {
                    variable = Some(name);
                }
                // Check for type
                if let Some(Token::Colon) = self.peek() {
                    self.advance(); // consume :
                    if let Some(Token::Identifier(name)) = self.advance().cloned() {
                        node_type = Some(name);
                    } else {
                        return Err("Expected type name after ':'".to_string());
                    }
                }
            }
            Some(Token::LBrace) => {
                // Properties only: ({prop: value})
            }
            _ => {}
        }

        // Check for properties
        if let Some(Token::LBrace) = self.peek() {
            properties = Some(self.parse_properties()?);
        }

        self.expect(&Token::RParen)?;

        Ok(NodePattern {
            variable,
            node_type,
            properties,
        })
    }

    /// Parse edge pattern: -[:TYPE]-> or <-[:TYPE]- or -[:TYPE]-
    fn parse_edge_pattern(&mut self) -> Result<EdgePattern, String> {
        let mut direction = EdgeDirection::Both;
        let mut incoming_start = false;

        // Check for incoming arrow start: <-
        if let Some(Token::LessThan) = self.peek() {
            self.advance(); // consume <
            incoming_start = true;
            direction = EdgeDirection::Incoming;
        }

        self.expect(&Token::Dash)?;

        // Parse the bracket part: [:TYPE {props}]
        self.expect(&Token::LBracket)?;

        let mut variable = None;
        let mut connection_type = None;
        let mut properties = None;

        // Check what comes next
        match self.peek() {
            Some(Token::RBracket) => {
                // Empty edge pattern: []
            }
            Some(Token::Colon) => {
                // No variable, just type: [:TYPE]
                self.advance(); // consume :
                if let Some(Token::Identifier(name)) = self.advance().cloned() {
                    connection_type = Some(name);
                } else {
                    return Err("Expected connection type after ':'".to_string());
                }
            }
            Some(Token::Identifier(_)) => {
                // Variable name
                if let Some(Token::Identifier(name)) = self.advance().cloned() {
                    variable = Some(name);
                }
                // Check for type
                if let Some(Token::Colon) = self.peek() {
                    self.advance(); // consume :
                    if let Some(Token::Identifier(name)) = self.advance().cloned() {
                        connection_type = Some(name);
                    } else {
                        return Err("Expected connection type after ':'".to_string());
                    }
                }
            }
            Some(Token::LBrace) => {
                // Properties only
            }
            _ => {}
        }

        // Check for properties
        if let Some(Token::LBrace) = self.peek() {
            properties = Some(self.parse_properties()?);
        }

        self.expect(&Token::RBracket)?;
        self.expect(&Token::Dash)?;

        // Check for outgoing arrow end: ->
        if let Some(Token::GreaterThan) = self.peek() {
            self.advance(); // consume >
            if incoming_start {
                // <-[]-> is invalid
                return Err("Invalid edge pattern: cannot have both < and >".to_string());
            }
            direction = EdgeDirection::Outgoing;
        } else if !incoming_start {
            // -[]- without direction is bidirectional
            direction = EdgeDirection::Both;
        }

        Ok(EdgePattern {
            variable,
            connection_type,
            direction,
            properties,
        })
    }

    /// Parse properties: {key: value, key2: value2}
    fn parse_properties(&mut self) -> Result<HashMap<String, PropertyMatcher>, String> {
        self.expect(&Token::LBrace)?;
        let mut props = HashMap::new();

        loop {
            match self.peek() {
                Some(Token::RBrace) => {
                    self.advance();
                    break;
                }
                Some(Token::Identifier(_)) => {
                    // Parse key: value
                    let key = if let Some(Token::Identifier(k)) = self.advance().cloned() {
                        k
                    } else {
                        return Err("Expected property key".to_string());
                    };

                    self.expect(&Token::Colon)?;

                    let value = self.parse_value()?;
                    props.insert(key, PropertyMatcher::Equals(value));

                    // Check for comma or end
                    if let Some(Token::Comma) = self.peek() {
                        self.advance();
                    }
                }
                _ => return Err("Expected property key or '}'".to_string()),
            }
        }

        Ok(props)
    }

    /// Parse a value (string, int, float, bool)
    fn parse_value(&mut self) -> Result<Value, String> {
        match self.advance().cloned() {
            Some(Token::StringLit(s)) => Ok(Value::String(s)),
            Some(Token::IntLit(i)) => Ok(Value::Int64(i)),
            Some(Token::FloatLit(f)) => Ok(Value::Float64(f)),
            Some(Token::BoolLit(b)) => Ok(Value::Boolean(b)),
            Some(token) => Err(format!("Expected value, got {:?}", token)),
            None => Err("Expected value, got end of input".to_string()),
        }
    }
}

pub fn parse_pattern(input: &str) -> Result<Pattern, String> {
    let tokens = tokenize(input)?;
    let mut parser = Parser::new(tokens);
    parser.parse_pattern()
}

// ============================================================================
// Executor
// ============================================================================

pub struct PatternExecutor<'a> {
    graph: &'a DirGraph,
    max_matches: Option<usize>,
}

impl<'a> PatternExecutor<'a> {
    pub fn new(graph: &'a DirGraph, max_matches: Option<usize>) -> Self {
        PatternExecutor { graph, max_matches }
    }

    /// Execute the pattern and return all matches
    pub fn execute(&self, pattern: &Pattern) -> Result<Vec<PatternMatch>, String> {
        if pattern.elements.is_empty() {
            return Ok(Vec::new());
        }

        // Start with the first node pattern
        let first_node = match &pattern.elements[0] {
            PatternElement::Node(np) => np,
            _ => return Err("Pattern must start with a node".to_string()),
        };

        // Find all nodes matching the first pattern
        let mut initial_nodes = self.find_matching_nodes(first_node)?;

        // Apply max_matches limit to initial nodes if this is a single-node pattern
        if pattern.elements.len() == 1 {
            if let Some(max) = self.max_matches {
                initial_nodes.truncate(max);
            }
        }

        // Initialize matches with first node bindings
        let mut matches: Vec<PatternMatch> = initial_nodes
            .iter()
            .map(|&idx| {
                let mut pm = PatternMatch {
                    bindings: HashMap::new(),
                };
                if let Some(ref var) = first_node.variable {
                    pm.bindings.insert(var.clone(), self.node_to_binding(idx));
                }
                pm
            })
            .collect();

        // Track current node indices for each match
        let mut current_indices: Vec<NodeIndex> = initial_nodes;

        // Process edge-node pairs
        let mut i = 1;
        while i < pattern.elements.len() {
            if self.max_matches.map_or(false, |max| matches.len() >= max) {
                break;
            }

            let edge_pattern = match &pattern.elements[i] {
                PatternElement::Edge(ep) => ep,
                _ => return Err("Expected edge pattern".to_string()),
            };

            i += 1;
            if i >= pattern.elements.len() {
                return Err("Edge pattern must be followed by node pattern".to_string());
            }

            let node_pattern = match &pattern.elements[i] {
                PatternElement::Node(np) => np,
                _ => return Err("Expected node pattern after edge".to_string()),
            };

            // Expand each current match
            let mut new_matches = Vec::new();
            let mut new_indices = Vec::new();

            for (_match_idx, (current_match, &source_idx)) in
                matches.iter().zip(current_indices.iter()).enumerate()
            {
                if self.max_matches.map_or(false, |max| new_matches.len() >= max) {
                    break;
                }

                // Find all valid expansions
                let expansions =
                    self.expand_from_node(source_idx, edge_pattern, node_pattern)?;

                for (target_idx, edge_binding) in expansions {
                    if self.max_matches.map_or(false, |max| new_matches.len() >= max) {
                        break;
                    }

                    let mut new_match = current_match.clone();

                    // Add edge binding if variable exists
                    if let Some(ref var) = edge_pattern.variable {
                        new_match.bindings.insert(var.clone(), edge_binding);
                    }

                    // Add node binding if variable exists
                    if let Some(ref var) = node_pattern.variable {
                        new_match
                            .bindings
                            .insert(var.clone(), self.node_to_binding(target_idx));
                    }

                    new_matches.push(new_match);
                    new_indices.push(target_idx);
                }
            }

            matches = new_matches;
            current_indices = new_indices;
            i += 1;
        }

        Ok(matches)
    }

    /// Find all nodes matching a node pattern
    fn find_matching_nodes(&self, pattern: &NodePattern) -> Result<Vec<NodeIndex>, String> {
        let candidates: Vec<NodeIndex> = if let Some(ref node_type) = pattern.node_type {
            // Use type index for O(1) lookup
            self.graph
                .type_indices
                .get(node_type)
                .cloned()
                .unwrap_or_default()
        } else {
            // No type specified - check all nodes
            self.graph.graph.node_indices().collect()
        };

        // Filter by properties if specified
        let filtered = if let Some(ref props) = pattern.properties {
            candidates
                .into_iter()
                .filter(|&idx| self.node_matches_properties(idx, props))
                .collect()
        } else {
            candidates
        };

        Ok(filtered)
    }

    /// Check if a node matches property filters
    fn node_matches_properties(
        &self,
        idx: NodeIndex,
        props: &HashMap<String, PropertyMatcher>,
    ) -> bool {
        if let Some(node) = self.graph.graph.node_weight(idx) {
            use crate::graph::schema::NodeData;
            match node {
                NodeData::Regular { properties, .. } => {
                    for (key, matcher) in props {
                        match properties.get(key) {
                            Some(value) => {
                                if !self.value_matches(value, matcher) {
                                    return false;
                                }
                            }
                            None => return false,
                        }
                    }
                    true
                }
                _ => false,
            }
        } else {
            false
        }
    }

    /// Check if a value matches a property matcher
    fn value_matches(&self, value: &Value, matcher: &PropertyMatcher) -> bool {
        match matcher {
            PropertyMatcher::Equals(expected) => value == expected,
        }
    }

    /// Expand from a source node via an edge pattern to nodes matching node pattern
    fn expand_from_node(
        &self,
        source: NodeIndex,
        edge_pattern: &EdgePattern,
        node_pattern: &NodePattern,
    ) -> Result<Vec<(NodeIndex, MatchBinding)>, String> {
        let mut results = Vec::new();

        // Determine which directions to check
        let directions = match edge_pattern.direction {
            EdgeDirection::Outgoing => vec![Direction::Outgoing],
            EdgeDirection::Incoming => vec![Direction::Incoming],
            EdgeDirection::Both => vec![Direction::Outgoing, Direction::Incoming],
        };

        for direction in directions {
            let edges = self.graph.graph.edges_directed(source, direction);

            for edge in edges {
                let edge_data = edge.weight();

                // Check connection type if specified
                if let Some(ref conn_type) = edge_pattern.connection_type {
                    if &edge_data.connection_type != conn_type {
                        continue;
                    }
                }

                // Check edge properties if specified
                if let Some(ref props) = edge_pattern.properties {
                    let matches = props.iter().all(|(key, matcher)| {
                        edge_data
                            .properties
                            .get(key)
                            .map(|v| self.value_matches(v, matcher))
                            .unwrap_or(false)
                    });
                    if !matches {
                        continue;
                    }
                }

                // Get target node
                let target = match direction {
                    Direction::Outgoing => edge.target(),
                    Direction::Incoming => edge.source(),
                };

                // Check if target matches node pattern
                if let Some(ref node_type) = node_pattern.node_type {
                    if let Some(node) = self.graph.graph.node_weight(target) {
                        use crate::graph::schema::NodeData;
                        match node {
                            NodeData::Regular { node_type: nt, .. } => {
                                if nt != node_type {
                                    continue;
                                }
                            }
                            _ => continue,
                        }
                    } else {
                        continue;
                    }
                }

                // Check node properties if specified
                if let Some(ref props) = node_pattern.properties {
                    if !self.node_matches_properties(target, props) {
                        continue;
                    }
                }

                // Create edge binding
                let edge_binding = MatchBinding::Edge {
                    source,
                    target,
                    connection_type: edge_data.connection_type.clone(),
                    properties: edge_data.properties.clone(),
                };

                results.push((target, edge_binding));
            }
        }

        Ok(results)
    }

    /// Convert a node to a binding
    fn node_to_binding(&self, idx: NodeIndex) -> MatchBinding {
        if let Some(node) = self.graph.graph.node_weight(idx) {
            use crate::graph::schema::NodeData;
            match node {
                NodeData::Regular {
                    node_type,
                    id,
                    title,
                    properties,
                } | NodeData::Schema {
                    node_type,
                    id,
                    title,
                    properties,
                } => {
                    let title_str = match title {
                        Value::String(s) => s.clone(),
                        Value::Int64(i) => i.to_string(),
                        Value::Float64(f) => f.to_string(),
                        Value::UniqueId(u) => u.to_string(),
                        _ => format!("{:?}", title),
                    };
                    MatchBinding::Node {
                        index: idx,
                        node_type: node_type.clone(),
                        title: title_str,
                        id: id.clone(),
                        properties: properties.clone(),
                    }
                },
            }
        } else {
            MatchBinding::Node {
                index: idx,
                node_type: "Unknown".to_string(),
                title: "Unknown".to_string(),
                id: Value::Null,
                properties: HashMap::new(),
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_simple() {
        let tokens = tokenize("(a:Person)").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::LParen,
                Token::Identifier("a".to_string()),
                Token::Colon,
                Token::Identifier("Person".to_string()),
                Token::RParen,
            ]
        );
    }

    #[test]
    fn test_tokenize_edge() {
        let tokens = tokenize("-[:KNOWS]->").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Dash,
                Token::LBracket,
                Token::Colon,
                Token::Identifier("KNOWS".to_string()),
                Token::RBracket,
                Token::Dash,
                Token::GreaterThan,
            ]
        );
    }

    #[test]
    fn test_tokenize_properties() {
        let tokens = tokenize("{name: \"Alice\", age: 30}").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::LBrace,
                Token::Identifier("name".to_string()),
                Token::Colon,
                Token::StringLit("Alice".to_string()),
                Token::Comma,
                Token::Identifier("age".to_string()),
                Token::Colon,
                Token::IntLit(30),
                Token::RBrace,
            ]
        );
    }

    #[test]
    fn test_parse_simple_node() {
        let pattern = parse_pattern("(p:Person)").unwrap();
        assert_eq!(pattern.elements.len(), 1);
        if let PatternElement::Node(np) = &pattern.elements[0] {
            assert_eq!(np.variable, Some("p".to_string()));
            assert_eq!(np.node_type, Some("Person".to_string()));
        } else {
            panic!("Expected node pattern");
        }
    }

    #[test]
    fn test_parse_node_with_properties() {
        let pattern = parse_pattern("(p:Person {name: \"Alice\"})").unwrap();
        if let PatternElement::Node(np) = &pattern.elements[0] {
            assert!(np.properties.is_some());
            let props = np.properties.as_ref().unwrap();
            assert!(props.contains_key("name"));
        } else {
            panic!("Expected node pattern");
        }
    }

    #[test]
    fn test_parse_single_hop() {
        let pattern = parse_pattern("(a:Person)-[:KNOWS]->(b:Person)").unwrap();
        assert_eq!(pattern.elements.len(), 3);

        if let PatternElement::Edge(ep) = &pattern.elements[1] {
            assert_eq!(ep.connection_type, Some("KNOWS".to_string()));
            assert_eq!(ep.direction, EdgeDirection::Outgoing);
        } else {
            panic!("Expected edge pattern");
        }
    }

    #[test]
    fn test_parse_incoming_edge() {
        let pattern = parse_pattern("(a:Person)<-[:KNOWS]-(b:Person)").unwrap();
        if let PatternElement::Edge(ep) = &pattern.elements[1] {
            assert_eq!(ep.direction, EdgeDirection::Incoming);
        } else {
            panic!("Expected edge pattern");
        }
    }

    #[test]
    fn test_parse_bidirectional_edge() {
        let pattern = parse_pattern("(a:Person)-[:KNOWS]-(b:Person)").unwrap();
        if let PatternElement::Edge(ep) = &pattern.elements[1] {
            assert_eq!(ep.direction, EdgeDirection::Both);
        } else {
            panic!("Expected edge pattern");
        }
    }

    #[test]
    fn test_parse_multi_hop() {
        let pattern =
            parse_pattern("(a:Person)-[:KNOWS]->(b:Person)-[:WORKS_AT]->(c:Company)").unwrap();
        assert_eq!(pattern.elements.len(), 5);
    }

    #[test]
    fn test_parse_anonymous_node() {
        let pattern = parse_pattern("(:Person)").unwrap();
        if let PatternElement::Node(np) = &pattern.elements[0] {
            assert_eq!(np.variable, None);
            assert_eq!(np.node_type, Some("Person".to_string()));
        } else {
            panic!("Expected node pattern");
        }
    }

    #[test]
    fn test_parse_empty_node() {
        let pattern = parse_pattern("()").unwrap();
        if let PatternElement::Node(np) = &pattern.elements[0] {
            assert_eq!(np.variable, None);
            assert_eq!(np.node_type, None);
        } else {
            panic!("Expected node pattern");
        }
    }
}
