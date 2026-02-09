// src/graph/cypher/parser.rs
// Cypher clause parser - delegates MATCH patterns to pattern_matching::parse_pattern()

use super::ast::*;
use super::tokenizer::CypherToken;
use crate::datatypes::values::Value;
use crate::graph::pattern_matching;

// ============================================================================
// Parser
// ============================================================================

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
            | Some(CypherToken::Detach) => true,
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
                    return Err("CREATE clause is not yet supported".to_string());
                }
                Some(CypherToken::Set) => {
                    return Err("SET clause is not yet supported".to_string());
                }
                Some(CypherToken::Delete) | Some(CypherToken::Detach) => {
                    return Err("DELETE clause is not yet supported".to_string());
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

        Ok(CypherQuery { clauses })
    }

    // ========================================================================
    // MATCH Clause
    // ========================================================================

    fn parse_match_clause(&mut self, optional: bool) -> Result<Clause, String> {
        self.expect(&CypherToken::Match)?;

        // Collect tokens until the next clause keyword, then reconstruct as a pattern string
        // for delegation to pattern_matching::parse_pattern()
        let patterns = self.parse_match_patterns()?;

        let clause = MatchClause { patterns };
        if optional {
            Ok(Clause::OptionalMatch(clause))
        } else {
            Ok(Clause::Match(clause))
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
        // Check for parenthesized predicate
        if self.check(&CypherToken::LParen) {
            // Could be a parenthesized predicate or the start of a pattern
            // Peek ahead to determine
            if self.looks_like_pattern_start() {
                // It's a pattern predicate - not yet supported
                return Err("Pattern predicates in WHERE not yet supported".to_string());
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
        // Pattern: (var:Type) or (:Type) or ()
        // Predicate: (expr op expr) or (NOT ...)
        match self.peek_at(1) {
            Some(CypherToken::RParen) => true, // ()
            Some(CypherToken::Colon) => true,  // (:Type)
            Some(CypherToken::Identifier(_)) => {
                // Could be (var:Type) or (expr ...)
                // Check if third token is Colon -> pattern, otherwise predicate
                matches!(self.peek_at(2), Some(CypherToken::Colon))
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
        if self.check(&CypherToken::Dash) {
            self.advance();
            let inner = self.parse_primary_expression()?;
            Ok(Expression::Negate(Box::new(inner)))
        } else {
            self.parse_primary_expression()
        }
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

            // List literal [...]
            Some(CypherToken::LBracket) => {
                self.advance();
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
                } else {
                    Ok(Expression::Variable(name))
                }
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
            match self.advance().cloned() {
                Some(CypherToken::Identifier(name)) => Some(name),
                _ => return Err("Expected alias name after AS".to_string()),
            }
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
        match self.advance().cloned() {
            Some(CypherToken::Identifier(alias)) => {
                Ok(Clause::Unwind(UnwindClause { expression, alias }))
            }
            _ => Err("Expected alias after UNWIND ... AS".to_string()),
        }
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
}
