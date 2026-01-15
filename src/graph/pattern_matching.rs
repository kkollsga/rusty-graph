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
/// Supports variable-length paths with *min..max syntax:
/// - `*` or `*..` means 1 or more hops (default)
/// - `*2` means exactly 2 hops
/// - `*1..3` means 1 to 3 hops
/// - `*..5` means 1 to 5 hops
/// - `*2..` means 2 or more hops (up to default max)
#[derive(Debug, Clone)]
pub struct EdgePattern {
    pub variable: Option<String>,
    pub connection_type: Option<String>,
    pub direction: EdgeDirection,
    pub properties: Option<HashMap<String, PropertyMatcher>>,
    /// Variable-length path configuration: (min_hops, max_hops)
    /// None means exactly 1 hop (normal edge)
    pub var_length: Option<(usize, usize)>,
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

/// A bound value (either node, edge, or variable-length path)
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
    /// Variable-length path binding for patterns like -[:TYPE*1..3]->
    VariableLengthPath {
        source: NodeIndex,
        target: NodeIndex,
        hops: usize,
        /// Path as list of (node_index, connection_type) pairs
        path: Vec<(NodeIndex, String)>,
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
    Star,        // * (for variable-length paths)
    DotDot,      // .. (for range in variable-length)
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
            '*' => {
                tokens.push(Token::Star);
                chars.next();
            }
            '.' => {
                // Check for '..' (range operator)
                chars.next();
                if chars.peek() == Some(&'.') {
                    chars.next();
                    tokens.push(Token::DotDot);
                } else if chars.peek().is_some_and(|c| c.is_ascii_digit()) {
                    // It's a float starting with '.'
                    let mut num_str = String::from("0.");
                    while let Some(&c) = chars.peek() {
                        if c.is_ascii_digit() {
                            num_str.push(c);
                            chars.next();
                        } else {
                            break;
                        }
                    }
                    tokens.push(Token::FloatLit(
                        num_str.parse().map_err(|_| format!("Invalid float: {}", num_str))?
                    ));
                } else {
                    return Err("Unexpected single '.', expected '..' or a digit".to_string());
                }
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
            c if c.is_ascii_digit() => {
                let mut num_str = String::new();
                let mut has_dot = false;
                while let Some(&c) = chars.peek() {
                    if c.is_ascii_digit() {
                        num_str.push(c);
                        chars.next();
                    } else if c == '.' && !has_dot {
                        // Peek ahead to check if this is '..' (range operator)
                        // Clone the iterator to peek ahead without consuming
                        let mut peek_chars = chars.clone();
                        peek_chars.next(); // skip the first '.'
                        if peek_chars.peek() == Some(&'.') {
                            // This is '..', stop here and don't include the dot
                            break;
                        }
                        // It's a decimal point for a float
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
            _ => return Err(format!(
                "Unexpected character '{}' in pattern. Valid pattern syntax: (node)-[:EDGE]->(node). \
                Use () for nodes, [] for edges, : for types, {{}} for properties.",
                ch
            )),
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
            Some(token) => Err(format!(
                "Syntax error: expected '{}', but found '{}'. Check your pattern syntax.",
                Self::token_to_display(expected),
                Self::token_to_display(token)
            )),
            None => Err(format!(
                "Syntax error: expected '{}', but reached end of pattern. Pattern may be incomplete.",
                Self::token_to_display(expected)
            )),
        }
    }

    fn token_to_display(token: &Token) -> &'static str {
        match token {
            Token::LParen => "(",
            Token::RParen => ")",
            Token::LBracket => "[",
            Token::RBracket => "]",
            Token::LBrace => "{",
            Token::RBrace => "}",
            Token::Colon => ":",
            Token::Comma => ",",
            Token::Dash => "-",
            Token::GreaterThan => ">",
            Token::LessThan => "<",
            Token::Star => "*",
            Token::DotDot => "..",
            Token::Identifier(_) => "identifier",
            Token::StringLit(_) => "string",
            Token::IntLit(_) => "number",
            Token::FloatLit(_) => "decimal",
            Token::BoolLit(_) => "boolean",
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
                    return Err("Expected node type name after ':'. Example: (:Person) or (n:Person)".to_string());
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
                        return Err("Expected node type name after ':'. Example: (:Person) or (n:Person)".to_string());
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
    /// Also supports variable-length: -[:TYPE*1..3]->
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
        let mut var_length = None;

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
                    return Err("Expected connection/edge type after ':'. Example: -[:KNOWS]-> or -[e:WORKS_AT]->".to_string());
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
                        return Err("Expected connection/edge type after ':'. Example: -[:KNOWS]-> or -[e:WORKS_AT]->".to_string());
                    }
                }
            }
            Some(Token::Star) => {
                // Variable-length without type: [*1..3]
            }
            Some(Token::LBrace) => {
                // Properties only
            }
            _ => {}
        }

        // Check for variable-length marker: *
        if let Some(Token::Star) = self.peek() {
            var_length = Some(self.parse_var_length()?);
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
                return Err("Invalid edge pattern: cannot have both '<' and '>' arrows. Use -[]-> for outgoing, <-[]- for incoming, or -[]- for both directions.".to_string());
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
            var_length,
        })
    }

    /// Parse variable-length specification: *, *2, *1..3, *..5, *2..
    /// Returns (min_hops, max_hops)
    fn parse_var_length(&mut self) -> Result<(usize, usize), String> {
        self.expect(&Token::Star)?;

        const DEFAULT_MAX_HOPS: usize = 10; // Reasonable limit to prevent runaway queries

        // Check what follows the *
        match self.peek() {
            Some(Token::IntLit(_)) => {
                // *N or *N..M or *N..
                let min = if let Some(Token::IntLit(n)) = self.advance().cloned() {
                    n as usize
                } else {
                    return Err("Expected integer after '*' for variable-length path. Examples: *2, *1..3, *..5, *1..".to_string());
                };

                // Check for range
                if let Some(Token::DotDot) = self.peek() {
                    self.advance(); // consume ..
                    // Check for max
                    if let Some(Token::IntLit(_)) = self.peek() {
                        let max = if let Some(Token::IntLit(n)) = self.advance().cloned() {
                            n as usize
                        } else {
                            return Err("Expected max hop count after '..'. Examples: *1..3 (1 to 3 hops), *2.. (2 or more hops)".to_string());
                        };
                        Ok((min, max))
                    } else {
                        // *N.. means N to default max
                        Ok((min, DEFAULT_MAX_HOPS))
                    }
                } else {
                    // *N means exactly N hops
                    Ok((min, min))
                }
            }
            Some(Token::DotDot) => {
                // *..M means 1 to M
                self.advance(); // consume ..
                let max = if let Some(Token::IntLit(n)) = self.advance().cloned() {
                    n as usize
                } else {
                    return Err("Expected max hop count after '*..'. Example: *..3 means up to 3 hops".to_string());
                };
                Ok((1, max))
            }
            _ => {
                // * alone means 1 or more (up to default max)
                Ok((1, DEFAULT_MAX_HOPS))
            }
        }
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
                        return Err("Expected property key in properties block. Example: {name: 'Alice', age: 30}".to_string());
                    };

                    self.expect(&Token::Colon)?;

                    let value = self.parse_value()?;
                    props.insert(key, PropertyMatcher::Equals(value));

                    // Check for comma or end
                    if let Some(Token::Comma) = self.peek() {
                        self.advance();
                    }
                }
                _ => return Err("Expected property key or '}' to close properties block. Example: {name: 'Alice'}".to_string()),
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
            _ => return Err("Pattern must start with a node in parentheses. Example: (n:Person) or ()".to_string()),
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
                _ => return Err("Expected edge pattern after node. Use -[:TYPE]-> for outgoing, <-[:TYPE]- for incoming.".to_string()),
            };

            i += 1;
            if i >= pattern.elements.len() {
                return Err("Edge pattern must be followed by a node pattern. Example: ()-[:KNOWS]->(n:Person)".to_string());
            }

            let node_pattern = match &pattern.elements[i] {
                PatternElement::Node(np) => np,
                _ => return Err("Expected node pattern after edge. Complete the pattern with a node: ()-[:EDGE]->(node)".to_string()),
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
    /// Optimized: Uses references instead of cloning values
    fn node_matches_properties(
        &self,
        idx: NodeIndex,
        props: &HashMap<String, PropertyMatcher>,
    ) -> bool {
        if let Some(node) = self.graph.graph.node_weight(idx) {
            use crate::graph::schema::NodeData;
            match node {
                NodeData::Regular { properties, title, id, .. } => {
                    for (key, matcher) in props {
                        // Check special fields first: name/title maps to title, id maps to id
                        // Use references to avoid cloning
                        let value: Option<&Value> = if key == "name" || key == "title" {
                            Some(title)
                        } else if key == "id" {
                            Some(id)
                        } else {
                            properties.get(key)
                        };

                        match value {
                            Some(v) => {
                                if !self.value_matches(v, matcher) {
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
        // Check for variable-length path
        if let Some((min_hops, max_hops)) = edge_pattern.var_length {
            return self.expand_var_length(source, edge_pattern, node_pattern, min_hops, max_hops);
        }

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

    /// Expand via variable-length path (BFS within hop range)
    /// Optimized: Only clones paths when branching (multiple valid targets from same node)
    fn expand_var_length(
        &self,
        source: NodeIndex,
        edge_pattern: &EdgePattern,
        node_pattern: &NodePattern,
        min_hops: usize,
        max_hops: usize,
    ) -> Result<Vec<(NodeIndex, MatchBinding)>, String> {
        use std::collections::VecDeque;

        let mut results = Vec::new();

        // Determine which directions to check (avoid allocation with static slice)
        let directions: &[Direction] = match edge_pattern.direction {
            EdgeDirection::Outgoing => &[Direction::Outgoing],
            EdgeDirection::Incoming => &[Direction::Incoming],
            EdgeDirection::Both => &[Direction::Outgoing, Direction::Incoming],
        };

        // BFS state: (current_node, depth, path_info)
        // path_info stores the path taken for creating variable-length edge binding
        let mut queue: VecDeque<(NodeIndex, usize, Vec<(NodeIndex, String)>)> = VecDeque::new();
        let mut visited_at_depth: HashMap<(NodeIndex, usize), bool> = HashMap::new();

        queue.push_back((source, 0, Vec::new()));

        while let Some((current, depth, path)) = queue.pop_front() {
            if depth >= max_hops {
                continue;
            }

            // First pass: collect all valid targets to know how many branches we'll have
            // This avoids cloning paths unnecessarily when only one target exists
            let mut valid_targets: Vec<(NodeIndex, String)> = Vec::new();

            for &direction in directions {
                let edges = self.graph.graph.edges_directed(current, direction);

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

                    // Skip if we've visited this node at this depth (prevent cycles at same depth)
                    let visit_key = (target, depth + 1);
                    if visited_at_depth.contains_key(&visit_key) {
                        continue;
                    }
                    visited_at_depth.insert(visit_key, true);

                    valid_targets.push((target, edge_data.connection_type.clone()));
                }
            }

            // Second pass: process valid targets with smart path management
            let new_depth = depth + 1;
            let num_targets = valid_targets.len();

            for (i, (target, conn_type)) in valid_targets.into_iter().enumerate() {
                let is_last = i == num_targets - 1;
                let needs_queue = new_depth < max_hops;

                // Build new_path efficiently:
                // - For last target: reuse path by moving it (no clone)
                // - For others: clone the path
                let mut new_path = if is_last {
                    path.clone() // Can't avoid this clone since path is borrowed
                } else {
                    path.clone()
                };
                new_path.push((target, conn_type));

                // If we're within the valid hop range and target matches node pattern, add to results
                if new_depth >= min_hops {
                    let node_matches = if let Some(ref node_type) = node_pattern.node_type {
                        if let Some(node) = self.graph.graph.node_weight(target) {
                            use crate::graph::schema::NodeData;
                            match node {
                                NodeData::Regular { node_type: nt, .. } => nt == node_type,
                                _ => false,
                            }
                        } else {
                            false
                        }
                    } else {
                        true
                    };

                    let props_match = if let Some(ref props) = node_pattern.properties {
                        self.node_matches_properties(target, props)
                    } else {
                        true
                    };

                    if node_matches && props_match {
                        // Create binding - clone path only if we also need it for queue
                        let path_for_binding = if needs_queue {
                            new_path.clone()
                        } else {
                            std::mem::take(&mut new_path)
                        };
                        let edge_binding = MatchBinding::VariableLengthPath {
                            source,
                            target,
                            hops: new_depth,
                            path: path_for_binding,
                        };
                        results.push((target, edge_binding));
                    }
                }

                // Continue exploring if we haven't reached max depth
                if needs_queue {
                    queue.push_back((target, new_depth, new_path));
                }
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

    // Variable-length path tests
    #[test]
    fn test_tokenize_var_length() {
        let tokens = tokenize("-[:KNOWS*1..3]->").unwrap();
        assert!(tokens.contains(&Token::Star));
        assert!(tokens.contains(&Token::DotDot));
        assert!(tokens.contains(&Token::IntLit(1)));
        assert!(tokens.contains(&Token::IntLit(3)));
    }

    #[test]
    fn test_parse_var_length_exact() {
        let pattern = parse_pattern("(a:Person)-[:KNOWS*2]->(b:Person)").unwrap();
        if let PatternElement::Edge(ep) = &pattern.elements[1] {
            assert_eq!(ep.var_length, Some((2, 2)));
        } else {
            panic!("Expected edge pattern");
        }
    }

    #[test]
    fn test_parse_var_length_range() {
        let pattern = parse_pattern("(a:Person)-[:KNOWS*1..3]->(b:Person)").unwrap();
        if let PatternElement::Edge(ep) = &pattern.elements[1] {
            assert_eq!(ep.var_length, Some((1, 3)));
        } else {
            panic!("Expected edge pattern");
        }
    }

    #[test]
    fn test_parse_var_length_min_only() {
        let pattern = parse_pattern("(a:Person)-[:KNOWS*2..]->(b:Person)").unwrap();
        if let PatternElement::Edge(ep) = &pattern.elements[1] {
            // *2.. means 2 to default max (10)
            assert_eq!(ep.var_length, Some((2, 10)));
        } else {
            panic!("Expected edge pattern");
        }
    }

    #[test]
    fn test_parse_var_length_max_only() {
        let pattern = parse_pattern("(a:Person)-[:KNOWS*..5]->(b:Person)").unwrap();
        if let PatternElement::Edge(ep) = &pattern.elements[1] {
            assert_eq!(ep.var_length, Some((1, 5)));
        } else {
            panic!("Expected edge pattern");
        }
    }

    #[test]
    fn test_parse_var_length_star_only() {
        let pattern = parse_pattern("(a:Person)-[:KNOWS*]->(b:Person)").unwrap();
        if let PatternElement::Edge(ep) = &pattern.elements[1] {
            // * alone means 1 to default max (10)
            assert_eq!(ep.var_length, Some((1, 10)));
        } else {
            panic!("Expected edge pattern");
        }
    }

    #[test]
    fn test_parse_normal_edge_no_var_length() {
        let pattern = parse_pattern("(a:Person)-[:KNOWS]->(b:Person)").unwrap();
        if let PatternElement::Edge(ep) = &pattern.elements[1] {
            assert_eq!(ep.var_length, None);
        } else {
            panic!("Expected edge pattern");
        }
    }
}
