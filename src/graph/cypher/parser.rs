// src/graph/cypher/parser.rs
// Cypher clause parser - delegates MATCH patterns to pattern_matching::parse_pattern()

use super::ast::*;
use super::tokenizer::{token_to_keyword_name, CypherToken};
use crate::datatypes::values::Value;
use crate::graph::pattern_matching;

// ============================================================================
// Parser
// ============================================================================

/// Tokenizes and parses Cypher query strings into a `CypherQuery` AST.
///
/// Handles the full Cypher clause set: MATCH, WHERE, RETURN, WITH,
/// ORDER BY, LIMIT, SKIP, CREATE, SET, DELETE, MERGE, REMOVE, UNWIND, UNION.
/// Uses a token-based recursive descent approach.
pub struct CypherParser {
    tokens: Vec<CypherToken>,
    pos: usize,
}

impl CypherParser {
    pub fn new(tokens: Vec<CypherToken>) -> Self {
        CypherParser { tokens, pos: 0 }
    }

    // ========================================================================
    // Token Navigation
    // ========================================================================

    fn peek(&self) -> Option<&CypherToken> {
        self.tokens.get(self.pos)
    }

    fn peek_at(&self, offset: usize) -> Option<&CypherToken> {
        self.tokens.get(self.pos + offset)
    }

    fn advance(&mut self) -> Option<&CypherToken> {
        let token = self.tokens.get(self.pos);
        if token.is_some() {
            self.pos += 1;
        }
        token
    }

    fn expect(&mut self, expected: &CypherToken) -> Result<(), String> {
        match self.peek() {
            Some(t) if t == expected => {
                self.advance();
                Ok(())
            }
            Some(t) => Err(format!("Expected {:?}, found {:?}", expected, t)),
            None => Err(format!("Expected {:?}, but reached end of query", expected)),
        }
    }

    fn has_tokens(&self) -> bool {
        self.pos < self.tokens.len()
    }

    /// Check if current position matches a keyword
    fn check(&self, token: &CypherToken) -> bool {
        self.peek() == Some(token)
    }

    /// Consume the next token as an alias name (after AS).
    /// Accepts identifiers and reserved keywords (e.g. `AS optional`, `AS type`).
    fn try_consume_alias_name(&mut self) -> Result<String, String> {
        match self.advance().cloned() {
            Some(CypherToken::Identifier(name)) => Ok(name),
            Some(ref token) => token_to_keyword_name(token)
                .ok_or_else(|| format!("Expected alias name after AS, got {:?}", token)),
            None => Err("Expected alias name after AS".to_string()),
        }
    }

    /// Check if we're at a clause boundary (start of a new clause)
    fn at_clause_boundary(&self) -> bool {
        match self.peek() {
            Some(CypherToken::Where)
            | Some(CypherToken::Return)
            | Some(CypherToken::With)
            | Some(CypherToken::Limit)
            | Some(CypherToken::Skip)
            | Some(CypherToken::Unwind)
            | Some(CypherToken::Union)
            | Some(CypherToken::Create)
            | Some(CypherToken::Set)
            | Some(CypherToken::Delete)
            | Some(CypherToken::Detach)
            | Some(CypherToken::Merge)
            | Some(CypherToken::Remove)
            | Some(CypherToken::On)
            | Some(CypherToken::Call)
            | Some(CypherToken::Yield) => true,
            Some(CypherToken::Match) => true,
            Some(CypherToken::Optional) => {
                // OPTIONAL MATCH
                self.peek_at(1) == Some(&CypherToken::Match)
            }
            Some(CypherToken::Order) => {
                // ORDER BY
                self.peek_at(1) == Some(&CypherToken::By)
            }
            None => true,
            _ => false,
        }
    }

    // ========================================================================
    // Top-Level Query Parser
    // ========================================================================

    pub fn parse_query(&mut self) -> Result<CypherQuery, String> {
        // Check for EXPLAIN prefix
        let explain = if self.check(&CypherToken::Explain) {
            self.advance();
            true
        } else {
            false
        };

        let mut clauses = Vec::new();

        while self.has_tokens() {
            // Skip semicolons between statements
            if self.check(&CypherToken::Semicolon) {
                self.advance();
                continue;
            }

            match self.peek() {
                Some(CypherToken::Match) => {
                    clauses.push(self.parse_match_clause(false)?);
                }
                Some(CypherToken::Optional) => {
                    // Check for OPTIONAL MATCH
                    if self.peek_at(1) == Some(&CypherToken::Match) {
                        self.advance(); // consume OPTIONAL
                        clauses.push(self.parse_match_clause(true)?);
                    } else {
                        return Err("Expected MATCH after OPTIONAL".to_string());
                    }
                }
                Some(CypherToken::Where) => {
                    clauses.push(self.parse_where_clause()?);
                }
                Some(CypherToken::Return) => {
                    clauses.push(self.parse_return_clause()?);
                }
                Some(CypherToken::With) => {
                    clauses.push(self.parse_with_clause()?);
                }
                Some(CypherToken::Order) => {
                    clauses.push(self.parse_order_by_clause()?);
                }
                Some(CypherToken::Limit) => {
                    clauses.push(self.parse_limit_clause()?);
                }
                Some(CypherToken::Skip) => {
                    clauses.push(self.parse_skip_clause()?);
                }
                Some(CypherToken::Unwind) => {
                    clauses.push(self.parse_unwind_clause()?);
                }
                Some(CypherToken::Union) => {
                    clauses.push(self.parse_union_clause()?);
                }
                Some(CypherToken::Create) => {
                    clauses.push(self.parse_create_clause()?);
                }
                Some(CypherToken::Set) => {
                    clauses.push(self.parse_set_clause()?);
                }
                Some(CypherToken::Delete) | Some(CypherToken::Detach) => {
                    clauses.push(self.parse_delete_clause()?);
                }
                Some(CypherToken::Remove) => {
                    clauses.push(self.parse_remove_clause()?);
                }
                Some(CypherToken::Merge) => {
                    clauses.push(self.parse_merge_clause()?);
                }
                Some(CypherToken::Call) => {
                    clauses.push(self.parse_call_clause()?);
                }
                Some(t) => {
                    return Err(format!("Unexpected token at start of clause: {:?}", t));
                }
                None => break,
            }
        }

        if clauses.is_empty() {
            return Err("Empty query".to_string());
        }

        Ok(CypherQuery { clauses, explain })
    }

    // ========================================================================
    // MATCH Clause
    // ========================================================================

    fn parse_match_clause(&mut self, optional: bool) -> Result<Clause, String> {
        self.expect(&CypherToken::Match)?;

        let mut path_assignments = Vec::new();

        // Check for path assignment: p = shortestPath(...)
        // Pattern: Identifier Equals [Identifier("shortestPath") LParen] pattern [RParen]
        if self.is_path_assignment() {
            let path_var = self.consume_identifier()?;
            self.expect(&CypherToken::Equals)?;

            // Check for shortestPath( wrapper
            let is_shortest = self.is_shortest_path_call();
            if is_shortest {
                self.advance(); // consume "shortestPath" identifier
                self.expect(&CypherToken::LParen)?;
            }

            let patterns = self.parse_match_patterns()?;

            if is_shortest {
                self.expect(&CypherToken::RParen)?;
            }

            path_assignments.push(PathAssignment {
                variable: path_var,
                pattern_index: 0,
                is_shortest_path: is_shortest,
            });

            let clause = MatchClause {
                patterns,
                path_assignments,
            };
            return if optional {
                Ok(Clause::OptionalMatch(clause))
            } else {
                Ok(Clause::Match(clause))
            };
        }

        // Normal MATCH clause
        let patterns = self.parse_match_patterns()?;

        let clause = MatchClause {
            patterns,
            path_assignments,
        };
        if optional {
            Ok(Clause::OptionalMatch(clause))
        } else {
            Ok(Clause::Match(clause))
        }
    }

    /// Check if current position looks like: Identifier = [shortestPath(] ...
    fn is_path_assignment(&self) -> bool {
        matches!(self.peek(), Some(CypherToken::Identifier(_)))
            && self.peek_at(1) == Some(&CypherToken::Equals)
    }

    /// Check if current position is shortestPath( — called AFTER consuming "var ="
    fn is_shortest_path_call(&self) -> bool {
        if let Some(CypherToken::Identifier(name)) = self.peek() {
            name.eq_ignore_ascii_case("shortestPath")
                && self.peek_at(1) == Some(&CypherToken::LParen)
        } else {
            false
        }
    }

    /// Consume an identifier token and return the string
    fn consume_identifier(&mut self) -> Result<String, String> {
        match self.advance() {
            Some(CypherToken::Identifier(s)) => Ok(s.clone()),
            other => Err(format!("Expected identifier, got {:?}", other)),
        }
    }

    /// Parse one or more comma-separated patterns in MATCH
    fn parse_match_patterns(&mut self) -> Result<Vec<pattern_matching::Pattern>, String> {
        let mut patterns = Vec::new();

        loop {
            // Reconstruct the pattern string from tokens until we hit a comma (at top-level)
            // or a clause boundary
            let pattern_str = self.extract_pattern_string()?;
            if pattern_str.is_empty() {
                return Err("Expected a pattern in MATCH clause".to_string());
            }

            let pattern = pattern_matching::parse_pattern(&pattern_str)
                .map_err(|e| format!("Pattern parse error: {}", e))?;
            patterns.push(pattern);

            // Check for comma to continue with more patterns
            if self.check(&CypherToken::Comma) {
                self.advance();
            } else {
                break;
            }
        }

        Ok(patterns)
    }

    /// Parse patterns inside EXISTS { ... } — same as parse_match_patterns but uses
    /// extract_exists_pattern_string which stops at RBrace instead of clause boundaries.
    fn parse_exists_patterns(&mut self) -> Result<Vec<pattern_matching::Pattern>, String> {
        let mut patterns = Vec::new();

        loop {
            let pattern_str = self.extract_exists_pattern_string()?;
            if pattern_str.is_empty() {
                if patterns.is_empty() {
                    return Err("Expected a pattern inside EXISTS { }".to_string());
                }
                break;
            }

            let pattern = pattern_matching::parse_pattern(&pattern_str)
                .map_err(|e| format!("Pattern parse error in EXISTS: {}", e))?;
            patterns.push(pattern);

            if self.check(&CypherToken::Comma) {
                self.advance();
            } else {
                break;
            }
        }

        Ok(patterns)
    }

    /// Extract tokens forming a pattern inside EXISTS { ... }, stopping at RBrace or comma.
    fn extract_exists_pattern_string(&mut self) -> Result<String, String> {
        // Skip optional MATCH keyword — standard Cypher allows EXISTS { MATCH (pattern) }
        if self.check(&CypherToken::Match) {
            self.advance();
        }

        let mut parts = Vec::new();
        let mut paren_depth = 0i32;
        let mut bracket_depth = 0i32;

        while self.has_tokens() {
            // Stop at closing brace (the EXISTS boundary)
            if paren_depth == 0 && bracket_depth == 0 && self.check(&CypherToken::RBrace) {
                break;
            }

            // Stop at comma at top level (pattern separator)
            if paren_depth == 0 && bracket_depth == 0 && self.check(&CypherToken::Comma) {
                break;
            }

            let token = self.advance().unwrap().clone();

            match &token {
                CypherToken::LParen => {
                    paren_depth += 1;
                    parts.push("(".to_string());
                }
                CypherToken::RParen => {
                    paren_depth -= 1;
                    parts.push(")".to_string());
                }
                CypherToken::LBracket => {
                    bracket_depth += 1;
                    parts.push("[".to_string());
                }
                CypherToken::RBracket => {
                    bracket_depth -= 1;
                    parts.push("]".to_string());
                }
                CypherToken::LBrace => parts.push("{".to_string()),
                CypherToken::RBrace => parts.push("}".to_string()),
                CypherToken::Colon => parts.push(":".to_string()),
                CypherToken::Comma => parts.push(",".to_string()),
                CypherToken::Dash => parts.push("-".to_string()),
                CypherToken::GreaterThan => parts.push(">".to_string()),
                CypherToken::LessThan => parts.push("<".to_string()),
                CypherToken::Star => parts.push("*".to_string()),
                CypherToken::DotDot => parts.push("..".to_string()),
                CypherToken::Dot => parts.push(".".to_string()),
                CypherToken::Identifier(s) => parts.push(s.clone()),
                CypherToken::StringLit(s) => parts.push(format!("'{}'", s)),
                CypherToken::IntLit(n) => parts.push(n.to_string()),
                CypherToken::FloatLit(f) => parts.push(f.to_string()),
                CypherToken::True => parts.push("true".to_string()),
                CypherToken::False => parts.push("false".to_string()),
                _ => {
                    return Err(format!("Unexpected token in EXISTS pattern: {:?}", token));
                }
            }
        }

        Ok(parts.join(" "))
    }

    /// Extract tokens forming a single pattern and reconstruct as a string
    /// for the existing pattern_matching parser.
    /// Stops at commas (outside parens/brackets), clause keywords, or end of input.
    fn extract_pattern_string(&mut self) -> Result<String, String> {
        let mut parts = Vec::new();
        let mut paren_depth = 0i32;
        let mut bracket_depth = 0i32;

        while self.has_tokens() {
            // Stop at clause boundaries (only at top level)
            if paren_depth == 0 && bracket_depth == 0 && self.at_clause_boundary() {
                break;
            }

            // Stop at comma at top level (pattern separator)
            if paren_depth == 0 && bracket_depth == 0 && self.check(&CypherToken::Comma) {
                break;
            }

            // Stop at AND/OR at top level (boolean operators in WHERE)
            if paren_depth == 0
                && bracket_depth == 0
                && (self.check(&CypherToken::And) || self.check(&CypherToken::Or))
            {
                break;
            }

            // Stop at RParen that would go negative (e.g. closing shortestPath(...))
            if paren_depth == 0 && self.check(&CypherToken::RParen) {
                break;
            }

            let token = self.advance().unwrap().clone();

            match &token {
                CypherToken::LParen => {
                    paren_depth += 1;
                    parts.push("(".to_string());
                }
                CypherToken::RParen => {
                    paren_depth -= 1;
                    parts.push(")".to_string());
                }
                CypherToken::LBracket => {
                    bracket_depth += 1;
                    parts.push("[".to_string());
                }
                CypherToken::RBracket => {
                    bracket_depth -= 1;
                    parts.push("]".to_string());
                }
                CypherToken::LBrace => parts.push("{".to_string()),
                CypherToken::RBrace => parts.push("}".to_string()),
                CypherToken::Colon => parts.push(":".to_string()),
                CypherToken::Comma => parts.push(",".to_string()),
                CypherToken::Dash => parts.push("-".to_string()),
                CypherToken::GreaterThan => parts.push(">".to_string()),
                CypherToken::LessThan => parts.push("<".to_string()),
                CypherToken::Star => parts.push("*".to_string()),
                CypherToken::DotDot => parts.push("..".to_string()),
                CypherToken::Dot => parts.push(".".to_string()),
                CypherToken::Identifier(s) => parts.push(s.clone()),
                CypherToken::StringLit(s) => parts.push(format!("'{}'", s)),
                CypherToken::IntLit(n) => parts.push(n.to_string()),
                CypherToken::FloatLit(f) => parts.push(f.to_string()),
                CypherToken::True => parts.push("true".to_string()),
                CypherToken::False => parts.push("false".to_string()),
                CypherToken::Parameter(name) => {
                    parts.push(format!("${}", name));
                }
                _ => {
                    return Err(format!("Unexpected token in MATCH pattern: {:?}", token));
                }
            }
        }

        Ok(parts.join(""))
    }

    // ========================================================================
    // WHERE Clause
    // ========================================================================

    fn parse_where_clause(&mut self) -> Result<Clause, String> {
        self.expect(&CypherToken::Where)?;
        let predicate = self.parse_predicate()?;
        Ok(Clause::Where(WhereClause { predicate }))
    }

    /// Parse predicate with OR as lowest precedence
    fn parse_predicate(&mut self) -> Result<Predicate, String> {
        self.parse_or_predicate()
    }

    /// Parse OR expressions (lowest precedence)
    fn parse_or_predicate(&mut self) -> Result<Predicate, String> {
        let mut left = self.parse_and_predicate()?;

        while self.check(&CypherToken::Or) {
            self.advance();
            let right = self.parse_and_predicate()?;
            left = Predicate::Or(Box::new(left), Box::new(right));
        }

        Ok(left)
    }

    /// Parse AND expressions
    fn parse_and_predicate(&mut self) -> Result<Predicate, String> {
        let mut left = self.parse_not_predicate()?;

        while self.check(&CypherToken::And) {
            self.advance();
            let right = self.parse_not_predicate()?;
            left = Predicate::And(Box::new(left), Box::new(right));
        }

        Ok(left)
    }

    /// Parse NOT expressions
    fn parse_not_predicate(&mut self) -> Result<Predicate, String> {
        if self.check(&CypherToken::Not) {
            self.advance();
            let inner = self.parse_not_predicate()?;
            Ok(Predicate::Not(Box::new(inner)))
        } else {
            self.parse_comparison_predicate()
        }
    }

    /// Parse comparison expressions and IS NULL/IS NOT NULL/IN
    fn parse_comparison_predicate(&mut self) -> Result<Predicate, String> {
        // Check for EXISTS { pattern }
        if self.check(&CypherToken::Exists) {
            self.advance(); // consume EXISTS
            if self.check(&CypherToken::LBrace) {
                self.advance(); // consume {
                let patterns = self.parse_exists_patterns()?;
                self.expect(&CypherToken::RBrace)?;
                return Ok(Predicate::Exists { patterns });
            } else if self.check(&CypherToken::LParen) {
                self.advance(); // consume outer (
                                // Support EXISTS((...)) — inner parens are the pattern
                if self.check(&CypherToken::LParen) {
                    let pattern_str = self.extract_pattern_string()?;
                    let pattern = crate::graph::pattern_matching::parse_pattern(&pattern_str)?;
                    self.expect(&CypherToken::RParen)?; // consume outer )
                    return Ok(Predicate::Exists {
                        patterns: vec![pattern],
                    });
                } else {
                    return Err("EXISTS(...) requires a pattern in parentheses, e.g. EXISTS((n)-[:REL]->())".to_string());
                }
            } else {
                return Err("Expected '{' or '(' after EXISTS".to_string());
            }
        }

        // Check for parenthesized predicate
        if self.check(&CypherToken::LParen) {
            // Could be a parenthesized predicate or the start of a pattern
            // Peek ahead to determine
            if self.looks_like_pattern_start() {
                // Inline pattern predicate — desugar to EXISTS { pattern }
                let pattern_str = self.extract_pattern_string()?;
                let pattern = crate::graph::pattern_matching::parse_pattern(&pattern_str)?;
                return Ok(Predicate::Exists {
                    patterns: vec![pattern],
                });
            }
            self.advance(); // consume (
            let pred = self.parse_predicate()?;
            self.expect(&CypherToken::RParen)?;
            return Ok(pred);
        }

        let left = self.parse_expression()?;

        // Check for IS NULL / IS NOT NULL
        if self.check(&CypherToken::Is) {
            self.advance(); // consume IS
            if self.check(&CypherToken::Not) {
                self.advance(); // consume NOT
                self.expect(&CypherToken::Null)?;
                return Ok(Predicate::IsNotNull(left));
            } else {
                self.expect(&CypherToken::Null)?;
                return Ok(Predicate::IsNull(left));
            }
        }

        // Check for IN
        if self.check(&CypherToken::In) {
            self.advance();
            let list = self.parse_list_expression()?;
            return Ok(Predicate::In { expr: left, list });
        }

        // Check for STARTS WITH
        if self.check(&CypherToken::StartsWith) {
            self.advance(); // consume STARTS
                            // Expect WITH keyword (we use With token)
            self.expect(&CypherToken::With)?;
            let pattern = self.parse_expression()?;
            return Ok(Predicate::StartsWith {
                expr: left,
                pattern,
            });
        }

        // Check for ENDS WITH
        if self.check(&CypherToken::EndsWith) {
            self.advance(); // consume ENDS
            self.expect(&CypherToken::With)?;
            let pattern = self.parse_expression()?;
            return Ok(Predicate::EndsWith {
                expr: left,
                pattern,
            });
        }

        // Check for CONTAINS
        if self.check(&CypherToken::Contains) {
            self.advance();
            let pattern = self.parse_expression()?;
            return Ok(Predicate::Contains {
                expr: left,
                pattern,
            });
        }

        // Check for comparison operator
        let operator = match self.peek() {
            Some(CypherToken::Equals) => ComparisonOp::Equals,
            Some(CypherToken::NotEquals) => ComparisonOp::NotEquals,
            Some(CypherToken::LessThan) => ComparisonOp::LessThan,
            Some(CypherToken::LessThanEquals) => ComparisonOp::LessThanEq,
            Some(CypherToken::GreaterThan) => ComparisonOp::GreaterThan,
            Some(CypherToken::GreaterThanEquals) => ComparisonOp::GreaterThanEq,
            Some(CypherToken::RegexMatch) => ComparisonOp::RegexMatch,
            _ => {
                // No operator - the expression itself is a boolean predicate
                // Convert expression to a comparison: expr <> false (truthy check)
                return Ok(Predicate::Comparison {
                    left: left.clone(),
                    operator: ComparisonOp::NotEquals,
                    right: Expression::Literal(Value::Boolean(false)),
                });
            }
        };

        self.advance(); // consume operator
        let right = self.parse_expression()?;

        Ok(Predicate::Comparison {
            left,
            operator,
            right,
        })
    }

    /// Parse a [value, value, ...] list for IN clause
    fn parse_list_expression(&mut self) -> Result<Vec<Expression>, String> {
        self.expect(&CypherToken::LBracket)?;
        let mut items = Vec::new();

        if !self.check(&CypherToken::RBracket) {
            items.push(self.parse_expression()?);
            while self.check(&CypherToken::Comma) {
                self.advance();
                items.push(self.parse_expression()?);
            }
        }

        self.expect(&CypherToken::RBracket)?;
        Ok(items)
    }

    /// Quick lookahead to check if ( starts a pattern (node pattern) vs a parenthesized predicate
    fn looks_like_pattern_start(&self) -> bool {
        // Pattern: (var:Type), (:Type), (), (var)-[...]->()
        // Predicate: (expr op expr), (NOT ...)
        match self.peek_at(1) {
            Some(CypherToken::RParen) => {
                // () or (var) closed immediately — pattern if followed by - or <
                matches!(
                    self.peek_at(2),
                    Some(CypherToken::Dash) | Some(CypherToken::LessThan)
                )
            }
            Some(CypherToken::Colon) => true, // (:Type)
            Some(CypherToken::Identifier(_)) => {
                match self.peek_at(2) {
                    Some(CypherToken::Colon) => true, // (var:Type
                    Some(CypherToken::RParen) => {
                        // (var) — pattern if followed by - or <  e.g. (p)-[:REL]->()
                        matches!(
                            self.peek_at(3),
                            Some(CypherToken::Dash) | Some(CypherToken::LessThan)
                        )
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }

    // ========================================================================
    // Expression Parser
    // ========================================================================

    /// Parse an expression with operator precedence:
    /// additive (+, -) < multiplicative (*, /) < unary (-) < primary
    fn parse_expression(&mut self) -> Result<Expression, String> {
        self.parse_additive_expression()
    }

    fn parse_additive_expression(&mut self) -> Result<Expression, String> {
        let mut left = self.parse_multiplicative_expression()?;

        loop {
            match self.peek() {
                Some(CypherToken::Plus) => {
                    self.advance();
                    let right = self.parse_multiplicative_expression()?;
                    left = Expression::Add(Box::new(left), Box::new(right));
                }
                Some(CypherToken::Dash) => {
                    // Dash could be subtraction or edge syntax - only treat as subtraction
                    // if we're in an expression context (not at clause boundary)
                    // Heuristic: if next token after dash is a number, identifier, or '(',
                    // it's subtraction. Otherwise, stop.
                    if self.peek_at(1).is_some_and(|t| {
                        matches!(
                            t,
                            CypherToken::IntLit(_)
                                | CypherToken::FloatLit(_)
                                | CypherToken::Identifier(_)
                                | CypherToken::LParen
                        )
                    }) {
                        // But check it's not an edge pattern (dash followed by bracket)
                        if self.peek_at(1) == Some(&CypherToken::LBracket) {
                            break;
                        }
                        self.advance();
                        let right = self.parse_multiplicative_expression()?;
                        left = Expression::Subtract(Box::new(left), Box::new(right));
                    } else {
                        break;
                    }
                }
                _ => break,
            }
        }

        Ok(left)
    }

    fn parse_multiplicative_expression(&mut self) -> Result<Expression, String> {
        let mut left = self.parse_unary_expression()?;

        loop {
            match self.peek() {
                Some(CypherToken::Star) => {
                    self.advance();
                    let right = self.parse_unary_expression()?;
                    left = Expression::Multiply(Box::new(left), Box::new(right));
                }
                Some(CypherToken::Slash) => {
                    self.advance();
                    let right = self.parse_unary_expression()?;
                    left = Expression::Divide(Box::new(left), Box::new(right));
                }
                _ => break,
            }
        }

        Ok(left)
    }

    fn parse_unary_expression(&mut self) -> Result<Expression, String> {
        let expr = if self.check(&CypherToken::Dash) {
            self.advance();
            let inner = self.parse_primary_expression()?;
            Expression::Negate(Box::new(inner))
        } else {
            self.parse_primary_expression()?
        };
        self.parse_postfix(expr)
    }

    /// Parse postfix operators: expr[index] or expr[start..end]
    fn parse_postfix(&mut self, mut expr: Expression) -> Result<Expression, String> {
        while self.check(&CypherToken::LBracket) {
            self.advance(); // consume [

            if self.check(&CypherToken::DotDot) {
                // [..end] — slice with no start
                self.advance(); // consume ..
                let end_expr = self.parse_expression()?;
                self.expect(&CypherToken::RBracket)?;
                expr = Expression::ListSlice {
                    expr: Box::new(expr),
                    start: None,
                    end: Some(Box::new(end_expr)),
                };
            } else {
                let first = self.parse_expression()?;
                if self.check(&CypherToken::DotDot) {
                    // [start..] or [start..end]
                    self.advance(); // consume ..
                    let end_expr = if self.check(&CypherToken::RBracket) {
                        None
                    } else {
                        Some(Box::new(self.parse_expression()?))
                    };
                    self.expect(&CypherToken::RBracket)?;
                    expr = Expression::ListSlice {
                        expr: Box::new(expr),
                        start: Some(Box::new(first)),
                        end: end_expr,
                    };
                } else {
                    // [index] — plain index access
                    self.expect(&CypherToken::RBracket)?;
                    expr = Expression::IndexAccess {
                        expr: Box::new(expr),
                        index: Box::new(first),
                    };
                }
            }
        }
        Ok(expr)
    }

    fn parse_primary_expression(&mut self) -> Result<Expression, String> {
        match self.peek().cloned() {
            // Numeric literals
            Some(CypherToken::IntLit(n)) => {
                self.advance();
                Ok(Expression::Literal(Value::Int64(n)))
            }
            Some(CypherToken::FloatLit(f)) => {
                self.advance();
                Ok(Expression::Literal(Value::Float64(f)))
            }

            // String literal
            Some(CypherToken::StringLit(s)) => {
                self.advance();
                Ok(Expression::Literal(Value::String(s)))
            }

            // Boolean literals
            Some(CypherToken::True) => {
                self.advance();
                Ok(Expression::Literal(Value::Boolean(true)))
            }
            Some(CypherToken::False) => {
                self.advance();
                Ok(Expression::Literal(Value::Boolean(false)))
            }

            // NULL literal
            Some(CypherToken::Null) => {
                self.advance();
                Ok(Expression::Literal(Value::Null))
            }

            // Star (for count(*))
            Some(CypherToken::Star) => {
                self.advance();
                Ok(Expression::Star)
            }

            // Parenthesized expression
            Some(CypherToken::LParen) => {
                self.advance();
                let expr = self.parse_expression()?;
                self.expect(&CypherToken::RParen)?;
                Ok(expr)
            }

            // List literal [...] or list comprehension [x IN list WHERE ... | expr]
            Some(CypherToken::LBracket) => {
                self.advance(); // consume [

                // Check for list comprehension: [x IN list ...]
                // Look for: Identifier IN
                if matches!(self.peek(), Some(CypherToken::Identifier(_)))
                    && self.peek_at(1) == Some(&CypherToken::In)
                {
                    return self.parse_list_comprehension();
                }

                // Otherwise: list literal [expr, expr, ...]
                let mut items = Vec::new();
                if !self.check(&CypherToken::RBracket) {
                    items.push(self.parse_expression()?);
                    while self.check(&CypherToken::Comma) {
                        self.advance();
                        items.push(self.parse_expression()?);
                    }
                }
                self.expect(&CypherToken::RBracket)?;
                Ok(Expression::ListLiteral(items))
            }

            // CASE expression
            Some(CypherToken::Case) => {
                self.advance();
                self.parse_case_expression()
            }

            // Parameter: $name
            Some(CypherToken::Parameter(name)) => {
                self.advance();
                Ok(Expression::Parameter(name))
            }

            // Identifier: could be variable, property access, or function call
            Some(CypherToken::Identifier(name)) => {
                self.advance();

                // Check for function call: identifier(
                if self.check(&CypherToken::LParen) {
                    return self.parse_function_call(name);
                }

                // Check for property access: identifier.property
                if self.check(&CypherToken::Dot) {
                    self.advance(); // consume dot
                    match self.advance().cloned() {
                        Some(CypherToken::Identifier(prop)) => Ok(Expression::PropertyAccess {
                            variable: name,
                            property: prop,
                        }),
                        _ => Err("Expected property name after '.'".to_string()),
                    }
                }
                // Check for map projection: identifier { .prop1, .prop2, alias: expr }
                else if self.check(&CypherToken::LBrace) {
                    self.parse_map_projection(name)
                } else {
                    Ok(Expression::Variable(name))
                }
            }

            // Keywords that can also be function names when followed by (
            Some(CypherToken::Contains) if self.peek_at(1) == Some(&CypherToken::LParen) => {
                self.advance();
                self.parse_function_call("contains".to_string())
            }

            Some(t) => Err(format!("Unexpected token in expression: {:?}", t)),
            None => Err("Unexpected end of query in expression".to_string()),
        }
    }

    /// Parse function call: name(args...)
    fn parse_function_call(&mut self, name: String) -> Result<Expression, String> {
        self.expect(&CypherToken::LParen)?;

        // Check for DISTINCT
        let distinct = if self.check(&CypherToken::Distinct) {
            self.advance();
            true
        } else {
            false
        };

        let mut args = Vec::new();

        if !self.check(&CypherToken::RParen) {
            args.push(self.parse_expression()?);
            while self.check(&CypherToken::Comma) {
                self.advance();
                args.push(self.parse_expression()?);
            }
        }

        self.expect(&CypherToken::RParen)?;

        Ok(Expression::FunctionCall {
            name,
            args,
            distinct,
        })
    }

    // ========================================================================
    // Map Projection
    // ========================================================================

    /// Parse map projection: variable { .prop1, .prop2, alias: expr }
    /// The variable name has already been consumed; LBrace is next.
    fn parse_map_projection(&mut self, variable: String) -> Result<Expression, String> {
        self.expect(&CypherToken::LBrace)?;

        let mut items = Vec::new();

        while !self.check(&CypherToken::RBrace) {
            if !items.is_empty() {
                self.expect(&CypherToken::Comma)?;
            }

            // Check for .property shorthand
            if self.check(&CypherToken::Dot) {
                self.advance(); // consume dot
                match self.advance().cloned() {
                    Some(CypherToken::Identifier(prop)) => {
                        items.push(MapProjectionItem::Property(prop));
                    }
                    _ => return Err("Expected property name after '.' in map projection".into()),
                }
            } else {
                // alias: expression
                let key = match self.advance().cloned() {
                    Some(CypherToken::Identifier(name)) => name,
                    other => {
                        return Err(format!(
                            "Expected property name or .property in map projection, got {:?}",
                            other
                        ))
                    }
                };
                self.expect(&CypherToken::Colon)?;
                let expr = self.parse_expression()?;
                items.push(MapProjectionItem::Alias { key, expr });
            }
        }

        self.expect(&CypherToken::RBrace)?;

        Ok(Expression::MapProjection { variable, items })
    }

    // ========================================================================
    // CASE Expression
    // ========================================================================

    /// Parse CASE expression (CASE token already consumed)
    /// Generic form: CASE WHEN predicate THEN result [WHEN ...] [ELSE default] END
    /// Simple form:  CASE operand WHEN value THEN result [WHEN ...] [ELSE default] END
    fn parse_case_expression(&mut self) -> Result<Expression, String> {
        // Determine form: if next token is WHEN, it's generic; otherwise parse operand
        let operand = if self.check(&CypherToken::When) {
            None
        } else {
            Some(Box::new(self.parse_expression()?))
        };

        let mut when_clauses = Vec::new();

        // Parse WHEN ... THEN ... pairs
        while self.check(&CypherToken::When) {
            self.advance(); // consume WHEN

            let condition = if operand.is_some() {
                // Simple form: WHEN value — compare against operand
                CaseCondition::Expression(self.parse_expression()?)
            } else {
                // Generic form: WHEN predicate — evaluated as boolean
                CaseCondition::Predicate(self.parse_predicate()?)
            };

            self.expect(&CypherToken::Then)?;
            let result = self.parse_expression()?;
            when_clauses.push((condition, result));
        }

        if when_clauses.is_empty() {
            return Err("CASE expression requires at least one WHEN clause".to_string());
        }

        // Optional ELSE
        let else_expr = if self.check(&CypherToken::Else) {
            self.advance();
            Some(Box::new(self.parse_expression()?))
        } else {
            None
        };

        self.expect(&CypherToken::End)?;

        Ok(Expression::Case {
            operand,
            when_clauses,
            else_expr,
        })
    }

    /// Parse list comprehension: x IN list_expr WHERE predicate | map_expr ]
    /// Opening [ already consumed.
    fn parse_list_comprehension(&mut self) -> Result<Expression, String> {
        // Variable name
        let variable = match self.advance() {
            Some(CypherToken::Identifier(name)) => name.clone(),
            _ => return Err("Expected variable name in list comprehension".to_string()),
        };

        self.expect(&CypherToken::In)?;
        let list_expr = self.parse_expression()?;

        // Optional WHERE filter
        let filter = if self.check(&CypherToken::Where) {
            self.advance();
            Some(Box::new(self.parse_predicate()?))
        } else {
            None
        };

        // Optional | map_expr
        let map_expr = if self.check(&CypherToken::Pipe) {
            self.advance();
            Some(Box::new(self.parse_expression()?))
        } else {
            None
        };

        self.expect(&CypherToken::RBracket)?;

        Ok(Expression::ListComprehension {
            variable,
            list_expr: Box::new(list_expr),
            filter,
            map_expr,
        })
    }

    // ========================================================================
    // RETURN Clause
    // ========================================================================

    fn parse_return_clause(&mut self) -> Result<Clause, String> {
        self.expect(&CypherToken::Return)?;

        let distinct = if self.check(&CypherToken::Distinct) {
            self.advance();
            true
        } else {
            false
        };

        let items = self.parse_return_items()?;

        Ok(Clause::Return(ReturnClause { items, distinct }))
    }

    /// Parse comma-separated return items: expr AS alias, expr AS alias, ...
    fn parse_return_items(&mut self) -> Result<Vec<ReturnItem>, String> {
        let mut items = Vec::new();
        items.push(self.parse_return_item()?);

        while self.check(&CypherToken::Comma) {
            self.advance();
            items.push(self.parse_return_item()?);
        }

        Ok(items)
    }

    fn parse_return_item(&mut self) -> Result<ReturnItem, String> {
        let expression = self.parse_expression()?;

        let alias = if self.check(&CypherToken::As) {
            self.advance();
            Some(self.try_consume_alias_name()?)
        } else {
            None
        };

        Ok(ReturnItem { expression, alias })
    }

    // ========================================================================
    // WITH Clause
    // ========================================================================

    fn parse_with_clause(&mut self) -> Result<Clause, String> {
        self.expect(&CypherToken::With)?;

        let distinct = if self.check(&CypherToken::Distinct) {
            self.advance();
            true
        } else {
            false
        };

        let items = self.parse_return_items()?;

        // Check for optional WHERE in WITH
        let where_clause = if self.check(&CypherToken::Where) {
            self.advance();
            Some(WhereClause {
                predicate: self.parse_predicate()?,
            })
        } else {
            None
        };

        Ok(Clause::With(WithClause {
            items,
            distinct,
            where_clause,
        }))
    }

    // ========================================================================
    // ORDER BY Clause
    // ========================================================================

    fn parse_order_by_clause(&mut self) -> Result<Clause, String> {
        self.expect(&CypherToken::Order)?;
        self.expect(&CypherToken::By)?;

        let mut items = Vec::new();
        items.push(self.parse_order_item()?);

        while self.check(&CypherToken::Comma) {
            self.advance();
            items.push(self.parse_order_item()?);
        }

        Ok(Clause::OrderBy(OrderByClause { items }))
    }

    fn parse_order_item(&mut self) -> Result<OrderItem, String> {
        let expression = self.parse_expression()?;

        let ascending = match self.peek() {
            Some(CypherToken::Asc) => {
                self.advance();
                true
            }
            Some(CypherToken::Desc) => {
                self.advance();
                false
            }
            _ => true, // default ascending
        };

        Ok(OrderItem {
            expression,
            ascending,
        })
    }

    // ========================================================================
    // LIMIT / SKIP
    // ========================================================================

    fn parse_limit_clause(&mut self) -> Result<Clause, String> {
        self.expect(&CypherToken::Limit)?;
        let count = self.parse_expression()?;
        Ok(Clause::Limit(LimitClause { count }))
    }

    fn parse_skip_clause(&mut self) -> Result<Clause, String> {
        self.expect(&CypherToken::Skip)?;
        let count = self.parse_expression()?;
        Ok(Clause::Skip(SkipClause { count }))
    }

    // ========================================================================
    // UNWIND / UNION (Phase 3 stubs)
    // ========================================================================

    fn parse_unwind_clause(&mut self) -> Result<Clause, String> {
        self.expect(&CypherToken::Unwind)?;
        let expression = self.parse_expression()?;
        self.expect(&CypherToken::As)?;
        let alias = self.try_consume_alias_name()?;
        Ok(Clause::Unwind(UnwindClause { expression, alias }))
    }

    fn parse_union_clause(&mut self) -> Result<Clause, String> {
        self.expect(&CypherToken::Union)?;
        let all = if self.check(&CypherToken::All) {
            self.advance();
            true
        } else {
            false
        };

        // Parse the rest as a new query
        let query = self.parse_query()?;

        Ok(Clause::Union(UnionClause {
            all,
            query: Box::new(query),
        }))
    }

    // ========================================================================
    // CREATE Clause
    // ========================================================================

    fn parse_create_clause(&mut self) -> Result<Clause, String> {
        self.expect(&CypherToken::Create)?;
        let mut patterns = Vec::new();

        loop {
            patterns.push(self.parse_create_pattern()?);
            if self.check(&CypherToken::Comma) {
                self.advance();
            } else {
                break;
            }
        }

        Ok(Clause::Create(CreateClause { patterns }))
    }

    /// Parse a single CREATE path pattern: (node)-[edge]->(node)...
    fn parse_create_pattern(&mut self) -> Result<CreatePattern, String> {
        let mut elements = Vec::new();
        elements.push(CreateElement::Node(self.parse_create_node()?));

        // Parse optional edge-node chains
        while matches!(
            self.peek(),
            Some(CypherToken::Dash) | Some(CypherToken::LessThan)
        ) {
            elements.push(CreateElement::Edge(self.parse_create_edge()?));
            elements.push(CreateElement::Node(self.parse_create_node()?));
        }

        Ok(CreatePattern { elements })
    }

    /// Parse a node in a CREATE pattern: (var:Label {key: expr, ...})
    fn parse_create_node(&mut self) -> Result<CreateNodePattern, String> {
        self.expect(&CypherToken::LParen)?;
        let mut variable = None;
        let mut label = None;
        let mut properties = Vec::new();

        // Parse optional variable name
        if let Some(CypherToken::Identifier(_)) = self.peek() {
            // It's a variable if followed by : or { or )
            // (not a property access or function call)
            if let Some(CypherToken::Identifier(name)) = self.peek().cloned() {
                self.advance();
                variable = Some(name);
            }
        }

        // Parse optional :Label
        if self.check(&CypherToken::Colon) {
            self.advance();
            if let Some(CypherToken::Identifier(name)) = self.peek().cloned() {
                self.advance();
                label = Some(name);
            } else {
                return Err("Expected label name after ':'".to_string());
            }
        }

        // Parse optional {key: expr, ...}
        if self.check(&CypherToken::LBrace) {
            properties = self.parse_create_properties()?;
        }

        self.expect(&CypherToken::RParen)?;
        Ok(CreateNodePattern {
            variable,
            label,
            properties,
        })
    }

    /// Parse CREATE properties: {key: expr, key: expr, ...}
    fn parse_create_properties(&mut self) -> Result<Vec<(String, Expression)>, String> {
        self.expect(&CypherToken::LBrace)?;
        let mut props = Vec::new();

        if !self.check(&CypherToken::RBrace) {
            loop {
                let key = match self.peek().cloned() {
                    Some(CypherToken::Identifier(k)) => {
                        self.advance();
                        k
                    }
                    other => {
                        return Err(format!("Expected property key, got {:?}", other));
                    }
                };
                self.expect(&CypherToken::Colon)?;
                let value_expr = self.parse_expression()?;
                props.push((key, value_expr));

                if self.check(&CypherToken::Comma) {
                    self.advance();
                } else {
                    break;
                }
            }
        }

        self.expect(&CypherToken::RBrace)?;
        Ok(props)
    }

    /// Parse an edge in a CREATE pattern: -[var:TYPE {props}]-> or <-[var:TYPE {props}]-
    fn parse_create_edge(&mut self) -> Result<CreateEdgePattern, String> {
        // Handle direction prefix: <- means incoming
        let incoming = if self.check(&CypherToken::LessThan) {
            self.advance();
            true
        } else {
            false
        };

        self.expect(&CypherToken::Dash)?;
        self.expect(&CypherToken::LBracket)?;

        let mut variable = None;
        let mut connection_type = None;
        let mut properties = Vec::new();

        // Parse optional variable name
        if let Some(CypherToken::Identifier(_)) = self.peek() {
            // Check if followed by : (variable:TYPE) or ] (just variable)
            if matches!(
                self.peek_at(1),
                Some(CypherToken::Colon) | Some(CypherToken::RBracket)
            ) {
                if let Some(CypherToken::Identifier(name)) = self.peek().cloned() {
                    self.advance();
                    variable = Some(name);
                }
            }
        }

        // Parse :TYPE (required for CREATE)
        if self.check(&CypherToken::Colon) {
            self.advance();
            if let Some(CypherToken::Identifier(name)) = self.peek().cloned() {
                self.advance();
                connection_type = Some(name);
            } else {
                return Err("Expected relationship type after ':'".to_string());
            }
        }

        let conn_type = connection_type
            .ok_or_else(|| "CREATE requires a relationship type (e.g. [:KNOWS])".to_string())?;

        // Parse optional properties
        if self.check(&CypherToken::LBrace) {
            properties = self.parse_create_properties()?;
        }

        self.expect(&CypherToken::RBracket)?;
        self.expect(&CypherToken::Dash)?;

        // Handle direction suffix
        let direction = if self.check(&CypherToken::GreaterThan) {
            self.advance();
            if incoming {
                return Err("Cannot have both < and > in CREATE edge pattern".to_string());
            }
            CreateEdgeDirection::Outgoing
        } else if incoming {
            CreateEdgeDirection::Incoming
        } else {
            return Err("CREATE edges must have a direction (-> or <-)".to_string());
        };

        Ok(CreateEdgePattern {
            variable,
            connection_type: conn_type,
            direction,
            properties,
        })
    }

    // ========================================================================
    // SET Clause
    // ========================================================================

    fn parse_set_clause(&mut self) -> Result<Clause, String> {
        self.expect(&CypherToken::Set)?;
        let items = self.parse_set_items()?;
        Ok(Clause::Set(SetClause { items }))
    }

    /// Parse comma-separated SET items (shared by SET and MERGE ON CREATE/ON MATCH)
    fn parse_set_items(&mut self) -> Result<Vec<SetItem>, String> {
        let mut items = Vec::new();

        loop {
            let var_name = match self.peek().cloned() {
                Some(CypherToken::Identifier(name)) => {
                    self.advance();
                    name
                }
                other => {
                    return Err(format!("Expected variable name in SET, got {:?}", other));
                }
            };

            if self.check(&CypherToken::Dot) {
                // Property assignment: var.prop = expr
                self.advance(); // consume .
                let prop_name = match self.peek().cloned() {
                    Some(CypherToken::Identifier(name)) => {
                        self.advance();
                        name
                    }
                    other => {
                        return Err(format!("Expected property name after '.', got {:?}", other));
                    }
                };
                self.expect(&CypherToken::Equals)?;
                let expression = self.parse_expression()?;
                items.push(SetItem::Property {
                    variable: var_name,
                    property: prop_name,
                    expression,
                });
            } else if self.check(&CypherToken::Colon) {
                // Label assignment: var:Label
                self.advance();
                let label = match self.peek().cloned() {
                    Some(CypherToken::Identifier(name)) => {
                        self.advance();
                        name
                    }
                    other => {
                        return Err(format!("Expected label name after ':', got {:?}", other));
                    }
                };
                items.push(SetItem::Label {
                    variable: var_name,
                    label,
                });
            } else {
                return Err("Expected '.' or ':' after variable name in SET".to_string());
            }

            if self.check(&CypherToken::Comma) {
                self.advance();
            } else {
                break;
            }
        }

        Ok(items)
    }

    // ========================================================================
    // DELETE Clause
    // ========================================================================

    fn parse_delete_clause(&mut self) -> Result<Clause, String> {
        let detach = if self.check(&CypherToken::Detach) {
            self.advance(); // consume DETACH
            true
        } else {
            false
        };
        self.expect(&CypherToken::Delete)?;

        let mut expressions = Vec::new();
        loop {
            let expr = match self.peek().cloned() {
                Some(CypherToken::Identifier(name)) => {
                    self.advance();
                    Expression::Variable(name)
                }
                other => {
                    return Err(format!("Expected variable name in DELETE, got {:?}", other));
                }
            };
            expressions.push(expr);

            if self.check(&CypherToken::Comma) {
                self.advance();
            } else {
                break;
            }
        }

        Ok(Clause::Delete(DeleteClause {
            detach,
            expressions,
        }))
    }

    // ========================================================================
    // REMOVE Clause
    // ========================================================================

    fn parse_remove_clause(&mut self) -> Result<Clause, String> {
        self.expect(&CypherToken::Remove)?;
        let mut items = Vec::new();

        loop {
            let var_name = match self.peek().cloned() {
                Some(CypherToken::Identifier(name)) => {
                    self.advance();
                    name
                }
                other => {
                    return Err(format!("Expected variable name in REMOVE, got {:?}", other));
                }
            };

            if self.check(&CypherToken::Dot) {
                // Property removal: var.prop
                self.advance(); // consume .
                let prop_name = match self.peek().cloned() {
                    Some(CypherToken::Identifier(name)) => {
                        self.advance();
                        name
                    }
                    other => {
                        return Err(format!(
                            "Expected property name after '.' in REMOVE, got {:?}",
                            other
                        ));
                    }
                };
                items.push(RemoveItem::Property {
                    variable: var_name,
                    property: prop_name,
                });
            } else if self.check(&CypherToken::Colon) {
                // Label removal: var:Label
                self.advance(); // consume :
                let label = match self.peek().cloned() {
                    Some(CypherToken::Identifier(name)) => {
                        self.advance();
                        name
                    }
                    other => {
                        return Err(format!(
                            "Expected label name after ':' in REMOVE, got {:?}",
                            other
                        ));
                    }
                };
                items.push(RemoveItem::Label {
                    variable: var_name,
                    label,
                });
            } else {
                return Err("Expected '.' or ':' after variable name in REMOVE".to_string());
            }

            if self.check(&CypherToken::Comma) {
                self.advance();
            } else {
                break;
            }
        }

        Ok(Clause::Remove(RemoveClause { items }))
    }

    // ========================================================================
    // MERGE Clause
    // ========================================================================

    fn parse_merge_clause(&mut self) -> Result<Clause, String> {
        self.expect(&CypherToken::Merge)?;
        let pattern = self.parse_create_pattern()?;

        let mut on_create = None;
        let mut on_match = None;

        // Parse optional ON CREATE SET / ON MATCH SET (can appear in either order)
        while self.check(&CypherToken::On) {
            self.advance(); // consume ON
            match self.peek() {
                Some(CypherToken::Create) => {
                    self.advance(); // consume CREATE
                    self.expect(&CypherToken::Set)?;
                    on_create = Some(self.parse_set_items()?);
                }
                Some(CypherToken::Match) => {
                    self.advance(); // consume MATCH
                    self.expect(&CypherToken::Set)?;
                    on_match = Some(self.parse_set_items()?);
                }
                other => {
                    return Err(format!(
                        "Expected CREATE or MATCH after ON in MERGE, got {:?}",
                        other
                    ));
                }
            }
        }

        Ok(Clause::Merge(MergeClause {
            pattern,
            on_create,
            on_match,
        }))
    }

    // ========================================================================
    // CALL Clause
    // ========================================================================

    fn parse_call_clause(&mut self) -> Result<Clause, String> {
        self.expect(&CypherToken::Call)?;

        // Parse procedure name
        let procedure_name = match self.peek().cloned() {
            Some(CypherToken::Identifier(name)) => {
                self.advance();
                name
            }
            other => {
                return Err(format!(
                    "Expected procedure name after CALL, got {:?}",
                    other
                ));
            }
        };

        // Parse argument list: ( [{key: val, ...}] )
        self.expect(&CypherToken::LParen)?;
        let parameters = if self.check(&CypherToken::LBrace) {
            self.parse_create_properties()?
        } else if !self.check(&CypherToken::RParen) {
            return Err(format!(
                "CALL parameters must use map syntax: CALL {}({{key: value, ...}}). \
                 Example: CALL {}({{damping_factor: 0.85}})",
                procedure_name, procedure_name
            ));
        } else {
            Vec::new()
        };
        self.expect(&CypherToken::RParen)?;

        // Parse YIELD clause (required)
        if !self.check(&CypherToken::Yield) {
            return Err(
                "CALL requires a YIELD clause, e.g. CALL pagerank() YIELD node, score".to_string(),
            );
        }
        self.advance(); // consume YIELD

        let yield_items = self.parse_yield_items()?;
        if yield_items.is_empty() {
            return Err("YIELD requires at least one column name".to_string());
        }

        Ok(Clause::Call(CallClause {
            procedure_name,
            parameters,
            yield_items,
        }))
    }

    /// Parse comma-separated YIELD items: name [AS alias], ...
    fn parse_yield_items(&mut self) -> Result<Vec<YieldItem>, String> {
        let mut items = Vec::new();

        loop {
            let name = match self.peek().cloned() {
                Some(CypherToken::Identifier(n)) => {
                    self.advance();
                    n
                }
                other => {
                    return Err(format!("Expected column name in YIELD, got {:?}", other));
                }
            };

            let alias = if self.check(&CypherToken::As) {
                self.advance();
                Some(self.try_consume_alias_name()?)
            } else {
                None
            };

            items.push(YieldItem { name, alias });

            if self.check(&CypherToken::Comma) {
                self.advance();
            } else {
                break;
            }
        }

        Ok(items)
    }
}

// ============================================================================
// Public API
// ============================================================================

/// Parse a Cypher query string into a CypherQuery AST
pub fn parse_cypher(input: &str) -> Result<CypherQuery, String> {
    let tokens = super::tokenizer::tokenize_cypher(input)?;
    let mut parser = CypherParser::new(tokens);
    parser.parse_query()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_match_return() {
        let query = parse_cypher("MATCH (n:Person) RETURN n").unwrap();
        assert_eq!(query.clauses.len(), 2);
        assert!(matches!(&query.clauses[0], Clause::Match(_)));
        assert!(matches!(&query.clauses[1], Clause::Return(_)));
    }

    #[test]
    fn test_match_where_return() {
        let query =
            parse_cypher("MATCH (n:Person) WHERE n.age > 30 RETURN n.name AS name").unwrap();
        assert_eq!(query.clauses.len(), 3);
        assert!(matches!(&query.clauses[0], Clause::Match(_)));
        assert!(matches!(&query.clauses[1], Clause::Where(_)));
        assert!(matches!(&query.clauses[2], Clause::Return(_)));

        // Check WHERE predicate
        if let Clause::Where(w) = &query.clauses[1] {
            if let Predicate::Comparison {
                left,
                operator,
                right,
            } = &w.predicate
            {
                assert!(
                    matches!(left, Expression::PropertyAccess { variable, property }
                    if variable == "n" && property == "age")
                );
                assert_eq!(*operator, ComparisonOp::GreaterThan);
                assert!(matches!(right, Expression::Literal(Value::Int64(30))));
            } else {
                panic!("Expected comparison predicate");
            }
        } else {
            panic!("Expected WHERE clause");
        }

        // Check RETURN alias
        if let Clause::Return(r) = &query.clauses[2] {
            assert_eq!(r.items.len(), 1);
            assert_eq!(r.items[0].alias, Some("name".to_string()));
        }
    }

    #[test]
    fn test_where_and_or() {
        let query = parse_cypher(
            "MATCH (n:Person) WHERE n.age > 18 AND n.city = 'Oslo' OR n.vip = true RETURN n",
        )
        .unwrap();

        if let Clause::Where(w) = &query.clauses[1] {
            // Should be: (age > 18 AND city = 'Oslo') OR vip = true
            assert!(matches!(&w.predicate, Predicate::Or(_, _)));
        }
    }

    #[test]
    fn test_where_not() {
        let query = parse_cypher("MATCH (n:Person) WHERE NOT n.active = false RETURN n").unwrap();

        if let Clause::Where(w) = &query.clauses[1] {
            assert!(matches!(&w.predicate, Predicate::Not(_)));
        }
    }

    #[test]
    fn test_where_is_null() {
        let query = parse_cypher("MATCH (n:Person) WHERE n.email IS NULL RETURN n").unwrap();

        if let Clause::Where(w) = &query.clauses[1] {
            assert!(matches!(&w.predicate, Predicate::IsNull(_)));
        }
    }

    #[test]
    fn test_where_is_not_null() {
        let query = parse_cypher("MATCH (n:Person) WHERE n.email IS NOT NULL RETURN n").unwrap();

        if let Clause::Where(w) = &query.clauses[1] {
            assert!(matches!(&w.predicate, Predicate::IsNotNull(_)));
        }
    }

    #[test]
    fn test_where_in_list() {
        let query = parse_cypher(
            "MATCH (n:Person) WHERE n.city IN ['Oslo', 'Bergen', 'Trondheim'] RETURN n",
        )
        .unwrap();

        if let Clause::Where(w) = &query.clauses[1] {
            if let Predicate::In { expr: _, list } = &w.predicate {
                assert_eq!(list.len(), 3);
            } else {
                panic!("Expected IN predicate");
            }
        }
    }

    #[test]
    fn test_return_multiple_items() {
        let query =
            parse_cypher("MATCH (n:Person) RETURN n.name AS name, n.age AS age, n.city").unwrap();

        if let Clause::Return(r) = &query.clauses[1] {
            assert_eq!(r.items.len(), 3);
            assert_eq!(r.items[0].alias, Some("name".to_string()));
            assert_eq!(r.items[1].alias, Some("age".to_string()));
            assert_eq!(r.items[2].alias, None);
        }
    }

    #[test]
    fn test_return_distinct() {
        let query = parse_cypher("MATCH (n:Person) RETURN DISTINCT n.city").unwrap();

        if let Clause::Return(r) = &query.clauses[1] {
            assert!(r.distinct);
        }
    }

    #[test]
    fn test_return_function_call() {
        let query = parse_cypher("MATCH (n:Person) RETURN count(n) AS total").unwrap();

        if let Clause::Return(r) = &query.clauses[1] {
            if let Expression::FunctionCall {
                name,
                args,
                distinct,
            } = &r.items[0].expression
            {
                assert_eq!(name, "count");
                assert_eq!(args.len(), 1);
                assert!(!distinct);
            } else {
                panic!("Expected function call");
            }
        }
    }

    #[test]
    fn test_return_count_star() {
        let query = parse_cypher("MATCH (n:Person) RETURN count(*) AS total").unwrap();

        if let Clause::Return(r) = &query.clauses[1] {
            if let Expression::FunctionCall { args, .. } = &r.items[0].expression {
                assert!(matches!(&args[0], Expression::Star));
            }
        }
    }

    #[test]
    fn test_return_count_distinct() {
        let query =
            parse_cypher("MATCH (n:Person) RETURN count(DISTINCT n.city) AS cities").unwrap();

        if let Clause::Return(r) = &query.clauses[1] {
            if let Expression::FunctionCall { distinct, .. } = &r.items[0].expression {
                assert!(distinct);
            }
        }
    }

    #[test]
    fn test_order_by_limit_skip() {
        let query =
            parse_cypher("MATCH (n:Person) RETURN n.name ORDER BY n.age DESC SKIP 5 LIMIT 10")
                .unwrap();

        assert!(matches!(&query.clauses[2], Clause::OrderBy(_)));
        assert!(matches!(&query.clauses[3], Clause::Skip(_)));
        assert!(matches!(&query.clauses[4], Clause::Limit(_)));

        if let Clause::OrderBy(o) = &query.clauses[2] {
            assert_eq!(o.items.len(), 1);
            assert!(!o.items[0].ascending);
        }
    }

    #[test]
    fn test_with_clause() {
        let query = parse_cypher(
            "MATCH (n:Person) WITH n.city AS city, count(n) AS cnt WHERE cnt > 5 RETURN city, cnt",
        )
        .unwrap();

        assert!(matches!(&query.clauses[1], Clause::With(_)));
        if let Clause::With(w) = &query.clauses[1] {
            assert_eq!(w.items.len(), 2);
            assert!(w.where_clause.is_some());
        }
    }

    #[test]
    fn test_optional_match() {
        let query =
            parse_cypher("MATCH (n:Person) OPTIONAL MATCH (n)-[:KNOWS]->(f:Person) RETURN n, f")
                .unwrap();

        assert!(matches!(&query.clauses[0], Clause::Match(_)));
        assert!(matches!(&query.clauses[1], Clause::OptionalMatch(_)));
        assert!(matches!(&query.clauses[2], Clause::Return(_)));
    }

    #[test]
    fn test_match_with_edge_pattern() {
        let query =
            parse_cypher("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name, b.name").unwrap();

        if let Clause::Match(m) = &query.clauses[0] {
            assert_eq!(m.patterns.len(), 1);
            assert_eq!(m.patterns[0].elements.len(), 3); // node, edge, node
        }
    }

    #[test]
    fn test_match_with_var_length() {
        let query = parse_cypher("MATCH (a:Person)-[:KNOWS*1..3]->(b:Person) RETURN a, b").unwrap();

        assert!(matches!(&query.clauses[0], Clause::Match(_)));
    }

    #[test]
    fn test_multiple_match_patterns() {
        let query = parse_cypher("MATCH (a:Person), (b:Company) RETURN a, b").unwrap();

        if let Clause::Match(m) = &query.clauses[0] {
            assert_eq!(m.patterns.len(), 2);
        }
    }

    #[test]
    fn test_case_insensitive() {
        let query = parse_cypher("match (n:Person) where n.age > 30 return n").unwrap();
        assert_eq!(query.clauses.len(), 3);
    }

    #[test]
    fn test_arithmetic_in_return() {
        let query =
            parse_cypher("MATCH (n:Product) RETURN n.price * 1.1 AS price_with_tax").unwrap();

        if let Clause::Return(r) = &query.clauses[1] {
            assert!(matches!(&r.items[0].expression, Expression::Multiply(_, _)));
        }
    }

    #[test]
    fn test_where_contains() {
        let query = parse_cypher("MATCH (n:Person) WHERE n.name CONTAINS 'son' RETURN n").unwrap();

        if let Clause::Where(w) = &query.clauses[1] {
            assert!(matches!(&w.predicate, Predicate::Contains { .. }));
        }
    }

    #[test]
    fn test_unwind() {
        let query = parse_cypher("UNWIND [1, 2, 3] AS x RETURN x").unwrap();

        assert!(matches!(&query.clauses[0], Clause::Unwind(_)));
        if let Clause::Unwind(u) = &query.clauses[0] {
            assert_eq!(u.alias, "x");
        }
    }

    #[test]
    fn test_case_generic_form() {
        let query = parse_cypher(
            "MATCH (n:Person) RETURN CASE WHEN n.age > 18 THEN 'adult' ELSE 'minor' END AS category",
        )
        .unwrap();

        if let Clause::Return(r) = &query.clauses[1] {
            assert!(
                matches!(&r.items[0].expression, Expression::Case { operand, .. } if operand.is_none())
            );
            assert_eq!(r.items[0].alias, Some("category".to_string()));
        } else {
            panic!("Expected RETURN clause");
        }
    }

    #[test]
    fn test_case_simple_form() {
        let query = parse_cypher(
            "MATCH (n:Person) RETURN CASE n.city WHEN 'Oslo' THEN 'capital' WHEN 'Bergen' THEN 'west' ELSE 'other' END",
        )
        .unwrap();

        if let Clause::Return(r) = &query.clauses[1] {
            if let Expression::Case {
                operand,
                when_clauses,
                else_expr,
            } = &r.items[0].expression
            {
                assert!(operand.is_some());
                assert_eq!(when_clauses.len(), 2);
                assert!(else_expr.is_some());
            } else {
                panic!("Expected CASE expression");
            }
        }
    }

    #[test]
    fn test_case_no_else() {
        let query =
            parse_cypher("MATCH (n:Person) RETURN CASE WHEN n.age > 18 THEN 'adult' END").unwrap();

        if let Clause::Return(r) = &query.clauses[1] {
            if let Expression::Case { else_expr, .. } = &r.items[0].expression {
                assert!(else_expr.is_none());
            } else {
                panic!("Expected CASE expression");
            }
        }
    }

    #[test]
    fn test_parameter_in_expression() {
        let query = parse_cypher("MATCH (n:Person) WHERE n.age > $min_age RETURN n.name").unwrap();

        if let Clause::Where(w) = &query.clauses[1] {
            if let Predicate::Comparison { right, .. } = &w.predicate {
                assert!(matches!(right, Expression::Parameter(name) if name == "min_age"));
            } else {
                panic!("Expected comparison predicate");
            }
        }
    }

    #[test]
    fn test_parameter_in_return() {
        let query = parse_cypher("MATCH (n:Person) RETURN n.name, $label AS label").unwrap();

        if let Clause::Return(r) = &query.clauses[1] {
            assert!(
                matches!(&r.items[1].expression, Expression::Parameter(name) if name == "label")
            );
        }
    }

    // ========================================================================
    // CREATE Clause
    // ========================================================================

    #[test]
    fn test_parse_create_node() {
        let query = parse_cypher("CREATE (n:Person {name: 'Alice', age: 30})").unwrap();
        assert_eq!(query.clauses.len(), 1);

        if let Clause::Create(c) = &query.clauses[0] {
            assert_eq!(c.patterns.len(), 1);
            assert_eq!(c.patterns[0].elements.len(), 1);
            if let CreateElement::Node(np) = &c.patterns[0].elements[0] {
                assert_eq!(np.variable, Some("n".to_string()));
                assert_eq!(np.label, Some("Person".to_string()));
                assert_eq!(np.properties.len(), 2);
                assert_eq!(np.properties[0].0, "name");
                assert_eq!(np.properties[1].0, "age");
            } else {
                panic!("Expected node element");
            }
        } else {
            panic!("Expected CREATE clause");
        }
    }

    #[test]
    fn test_parse_create_edge() {
        let query = parse_cypher("MATCH (a:Person), (b:Person) CREATE (a)-[:KNOWS]->(b)").unwrap();
        assert_eq!(query.clauses.len(), 2);
        assert!(matches!(&query.clauses[0], Clause::Match(_)));
        assert!(matches!(&query.clauses[1], Clause::Create(_)));

        if let Clause::Create(c) = &query.clauses[1] {
            assert_eq!(c.patterns[0].elements.len(), 3); // node, edge, node
            if let CreateElement::Edge(ep) = &c.patterns[0].elements[1] {
                assert_eq!(ep.connection_type, "KNOWS");
                assert_eq!(ep.direction, CreateEdgeDirection::Outgoing);
            } else {
                panic!("Expected edge element");
            }
        }
    }

    #[test]
    fn test_parse_create_path() {
        let query =
            parse_cypher("CREATE (a:Person {name: 'A'})-[:KNOWS]->(b:Person {name: 'B'})").unwrap();

        if let Clause::Create(c) = &query.clauses[0] {
            assert_eq!(c.patterns[0].elements.len(), 3);
            assert!(matches!(&c.patterns[0].elements[0], CreateElement::Node(_)));
            assert!(matches!(&c.patterns[0].elements[1], CreateElement::Edge(_)));
            assert!(matches!(&c.patterns[0].elements[2], CreateElement::Node(_)));
        }
    }

    #[test]
    fn test_parse_create_with_params() {
        let query = parse_cypher("CREATE (n:Person {name: $name, age: $age})").unwrap();

        if let Clause::Create(c) = &query.clauses[0] {
            if let CreateElement::Node(np) = &c.patterns[0].elements[0] {
                assert!(matches!(&np.properties[0].1, Expression::Parameter(n) if n == "name"));
                assert!(matches!(&np.properties[1].1, Expression::Parameter(n) if n == "age"));
            }
        }
    }

    #[test]
    fn test_parse_create_incoming_edge() {
        let query =
            parse_cypher("MATCH (a:Person), (b:Person) CREATE (a)<-[:FOLLOWS]-(b)").unwrap();

        if let Clause::Create(c) = &query.clauses[1] {
            if let CreateElement::Edge(ep) = &c.patterns[0].elements[1] {
                assert_eq!(ep.connection_type, "FOLLOWS");
                assert_eq!(ep.direction, CreateEdgeDirection::Incoming);
            }
        }
    }

    // ========================================================================
    // SET Clause
    // ========================================================================

    #[test]
    fn test_parse_set_property() {
        let query = parse_cypher("MATCH (n:Person) SET n.age = 31").unwrap();
        assert_eq!(query.clauses.len(), 2);
        assert!(matches!(&query.clauses[1], Clause::Set(_)));

        if let Clause::Set(s) = &query.clauses[1] {
            assert_eq!(s.items.len(), 1);
            if let SetItem::Property {
                variable,
                property,
                expression,
            } = &s.items[0]
            {
                assert_eq!(variable, "n");
                assert_eq!(property, "age");
                assert!(matches!(expression, Expression::Literal(Value::Int64(31))));
            }
        }
    }

    #[test]
    fn test_parse_set_multiple() {
        let query = parse_cypher("MATCH (n:Person) SET n.age = 31, n.city = 'Bergen'").unwrap();

        if let Clause::Set(s) = &query.clauses[1] {
            assert_eq!(s.items.len(), 2);
            if let SetItem::Property { property, .. } = &s.items[0] {
                assert_eq!(property, "age");
            }
            if let SetItem::Property { property, .. } = &s.items[1] {
                assert_eq!(property, "city");
            }
        }
    }

    #[test]
    fn test_parse_set_expression() {
        let query = parse_cypher("MATCH (n:Person) SET n.salary = n.salary * 1.1").unwrap();

        if let Clause::Set(s) = &query.clauses[1] {
            if let SetItem::Property { expression, .. } = &s.items[0] {
                assert!(matches!(expression, Expression::Multiply(_, _)));
            }
        }
    }

    #[test]
    fn test_parse_match_create_set_return() {
        let query = parse_cypher(
            "MATCH (a:Person) CREATE (a)-[:RATED]->(r:Review {text: 'Great'}) SET a.reviews = a.reviews + 1 RETURN a, r",
        ).unwrap();

        assert_eq!(query.clauses.len(), 4);
        assert!(matches!(&query.clauses[0], Clause::Match(_)));
        assert!(matches!(&query.clauses[1], Clause::Create(_)));
        assert!(matches!(&query.clauses[2], Clause::Set(_)));
        assert!(matches!(&query.clauses[3], Clause::Return(_)));
    }

    // ========================================================================
    // DELETE Clause
    // ========================================================================

    #[test]
    fn test_parse_delete() {
        let query = parse_cypher("MATCH (n:Person) DELETE n").unwrap();
        assert_eq!(query.clauses.len(), 2);
        if let Clause::Delete(d) = &query.clauses[1] {
            assert!(!d.detach);
            assert_eq!(d.expressions.len(), 1);
            assert!(matches!(&d.expressions[0], Expression::Variable(v) if v == "n"));
        } else {
            panic!("Expected DELETE clause");
        }
    }

    #[test]
    fn test_parse_detach_delete() {
        let query = parse_cypher("MATCH (n:Person) DETACH DELETE n").unwrap();
        if let Clause::Delete(d) = &query.clauses[1] {
            assert!(d.detach);
            assert_eq!(d.expressions.len(), 1);
        } else {
            panic!("Expected DELETE clause");
        }
    }

    #[test]
    fn test_parse_delete_multiple() {
        let query = parse_cypher("MATCH (a)-[r]->(b) DELETE a, r, b").unwrap();
        if let Clause::Delete(d) = &query.clauses[1] {
            assert_eq!(d.expressions.len(), 3);
        }
    }

    // ========================================================================
    // REMOVE Clause
    // ========================================================================

    #[test]
    fn test_parse_remove_property() {
        let query = parse_cypher("MATCH (n:Person) REMOVE n.age").unwrap();
        assert!(matches!(&query.clauses[1], Clause::Remove(_)));
        if let Clause::Remove(r) = &query.clauses[1] {
            assert_eq!(r.items.len(), 1);
            if let RemoveItem::Property { variable, property } = &r.items[0] {
                assert_eq!(variable, "n");
                assert_eq!(property, "age");
            } else {
                panic!("Expected property removal");
            }
        }
    }

    #[test]
    fn test_parse_remove_multiple() {
        let query = parse_cypher("MATCH (n:Person) REMOVE n.age, n.city").unwrap();
        if let Clause::Remove(r) = &query.clauses[1] {
            assert_eq!(r.items.len(), 2);
        }
    }

    #[test]
    fn test_parse_remove_label() {
        let query = parse_cypher("MATCH (n:Person) REMOVE n:Temporary").unwrap();
        if let Clause::Remove(r) = &query.clauses[1] {
            assert!(
                matches!(&r.items[0], RemoveItem::Label { variable, label } if variable == "n" && label == "Temporary")
            );
        }
    }

    // ========================================================================
    // MERGE Clause
    // ========================================================================

    #[test]
    fn test_parse_merge_node() {
        let query = parse_cypher("MERGE (n:Person {name: 'Alice'})").unwrap();
        assert_eq!(query.clauses.len(), 1);
        assert!(matches!(&query.clauses[0], Clause::Merge(_)));
        if let Clause::Merge(m) = &query.clauses[0] {
            assert_eq!(m.pattern.elements.len(), 1);
            assert!(m.on_create.is_none());
            assert!(m.on_match.is_none());
        }
    }

    #[test]
    fn test_parse_merge_on_create() {
        let query =
            parse_cypher("MERGE (n:Person {name: 'Alice'}) ON CREATE SET n.age = 30").unwrap();
        if let Clause::Merge(m) = &query.clauses[0] {
            assert!(m.on_create.is_some());
            assert!(m.on_match.is_none());
            assert_eq!(m.on_create.as_ref().unwrap().len(), 1);
        }
    }

    #[test]
    fn test_parse_merge_on_match() {
        let query =
            parse_cypher("MERGE (n:Person {name: 'Alice'}) ON MATCH SET n.visits = 1").unwrap();
        if let Clause::Merge(m) = &query.clauses[0] {
            assert!(m.on_create.is_none());
            assert!(m.on_match.is_some());
        }
    }

    #[test]
    fn test_parse_merge_both() {
        let query = parse_cypher(
            "MERGE (n:Person {name: 'Alice'}) ON CREATE SET n.age = 30 ON MATCH SET n.visits = 1",
        )
        .unwrap();
        if let Clause::Merge(m) = &query.clauses[0] {
            assert!(m.on_create.is_some());
            assert!(m.on_match.is_some());
        }
    }

    #[test]
    fn test_parse_merge_relationship() {
        let query = parse_cypher("MATCH (a:Person), (b:Person) MERGE (a)-[r:KNOWS]->(b)").unwrap();
        assert_eq!(query.clauses.len(), 2);
        if let Clause::Merge(m) = &query.clauses[1] {
            assert_eq!(m.pattern.elements.len(), 3);
        }
    }

    #[test]
    fn test_reserved_word_as_alias() {
        // Keywords should be valid alias names after AS
        for keyword in &[
            "optional", "match", "where", "return", "order", "limit", "type", "set", "all",
            "distinct", "contains", "exists", "null", "true", "false", "in", "is", "not",
        ] {
            let query_str = format!("MATCH (n) RETURN n AS {}", keyword);
            let query = parse_cypher(&query_str)
                .unwrap_or_else(|e| panic!("Failed to parse 'RETURN n AS {}': {}", keyword, e));
            if let Clause::Return(ret) = &query.clauses[1] {
                assert_eq!(
                    ret.items[0].alias.as_deref(),
                    Some(*keyword),
                    "Alias should be '{}' for keyword",
                    keyword
                );
            } else {
                panic!("Expected RETURN clause");
            }
        }
    }

    #[test]
    fn test_reserved_word_as_unwind_alias() {
        let query = parse_cypher("UNWIND [1,2] AS optional").unwrap();
        if let Clause::Unwind(u) = &query.clauses[0] {
            assert_eq!(u.alias, "optional");
        } else {
            panic!("Expected UNWIND clause");
        }
    }

    #[test]
    fn test_reserved_word_as_yield_alias() {
        let query = parse_cypher("CALL pagerank() YIELD node AS optional, score AS limit").unwrap();
        if let Clause::Call(c) = &query.clauses[0] {
            assert_eq!(c.yield_items[0].alias.as_deref(), Some("optional"));
            assert_eq!(c.yield_items[1].alias.as_deref(), Some("limit"));
        } else {
            panic!("Expected CALL clause");
        }
    }
}
