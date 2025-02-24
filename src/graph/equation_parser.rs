use std::collections::HashMap;
use crate::datatypes::values::Value;

#[derive(Debug, Clone)]
pub enum Expr {
    Number(f64),
    Variable(String),
    Add(Box<Expr>, Box<Expr>),
    Subtract(Box<Expr>, Box<Expr>),
    Multiply(Box<Expr>, Box<Expr>),
    Divide(Box<Expr>, Box<Expr>),
    Aggregate(AggregateType, Box<Expr>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum AggregateType {
    Sum,
    Mean,
    Std,
    Min,
    Max,
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Number(f64),
    Identifier(String),
    Plus,
    Minus,
    Star,
    Slash,
    LParen,
    RParen,
    Aggregate(AggregateType),
}

pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    pub fn parse_expression(input: &str) -> Result<Expr, String> {
        let mut parser = Self::new(input);
        parser.parse_aggregate()
    }

    fn new(input: &str) -> Self {
        let tokens = Self::tokenize(input);
        Parser {
            tokens,
            pos: 0,
        }
    }

    fn tokenize(input: &str) -> Vec<Token> {
        let mut tokens = Vec::new();
        let mut chars = input.chars().peekable();
        
        while let Some(&c) = chars.peek() {
            match c {
                '0'..='9' | '.' => {
                    let mut number = String::new();
                    let mut has_decimal = false;
                    
                    while let Some(&c) = chars.peek() {
                        match c {
                            '0'..='9' => {
                                number.push(c);
                                chars.next();
                            }
                            '.' if !has_decimal => {
                                has_decimal = true;
                                number.push(c);
                                chars.next();
                            }
                            _ => break,
                        }
                    }
                    
                    if let Ok(num) = number.parse() {
                        tokens.push(Token::Number(num));
                    }
                }
                'a'..='z' | 'A'..='Z' | '_' => {
                    let mut ident = String::new();
                    while let Some(&c) = chars.peek() {
                        if c.is_alphanumeric() || c == '_' {
                            ident.push(c);
                            chars.next();
                        } else {
                            break;
                        }
                    }
                    match ident.as_str() {
                        "sum" => tokens.push(Token::Aggregate(AggregateType::Sum)),
                        "mean" | "avg" => tokens.push(Token::Aggregate(AggregateType::Mean)),
                        "std" => tokens.push(Token::Aggregate(AggregateType::Std)),
                        "min" => tokens.push(Token::Aggregate(AggregateType::Min)),
                        "max" => tokens.push(Token::Aggregate(AggregateType::Max)),
                        _ => tokens.push(Token::Identifier(ident)),
                    }
                }
                '+' => {
                    tokens.push(Token::Plus);
                    chars.next();
                }
                '-' => {
                    tokens.push(Token::Minus);
                    chars.next();
                }
                '*' => {
                    tokens.push(Token::Star);
                    chars.next();
                }
                '/' => {
                    tokens.push(Token::Slash);
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
                ' ' | '\t' | '\n' | '\r' => {
                    chars.next();
                }
                _ => return vec![],  // Return empty vec for invalid input
            }
        }
        tokens
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn consume(&mut self) -> Option<Token> {
        if self.pos < self.tokens.len() {
            self.pos += 1;
            Some(self.tokens[self.pos - 1].clone())
        } else {
            None
        }
    }

    fn peek_and_consume_if<F>(&mut self, predicate: F) -> Option<Token>
    where
        F: FnOnce(&Token) -> bool,
    {
        match self.peek() {
            Some(token) if predicate(token) => self.consume(),
            _ => None,
        }
    }

    fn parse_aggregate(&mut self) -> Result<Expr, String> {
        let token = self.peek_and_consume_if(|t| matches!(t, Token::Aggregate(_)));
        
        match token {
            Some(Token::Aggregate(agg_type)) => {
                // Expect opening parenthesis
                if self.peek_and_consume_if(|t| matches!(t, Token::LParen)).is_none() {
                    return Err("Expected opening parenthesis after aggregate function".to_string());
                }

                // Parse the inner expression
                let expr = self.parse_expr()?;

                // Expect closing parenthesis
                if self.peek_and_consume_if(|t| matches!(t, Token::RParen)).is_none() {
                    return Err("Expected closing parenthesis after aggregate expression".to_string());
                }

                Ok(Expr::Aggregate(agg_type, Box::new(expr)))
            }
            _ => self.parse_expr()
        }
    }

    fn parse_expr(&mut self) -> Result<Expr, String> {
        let mut left = self.parse_term()?;

        while let Some(token) = self.peek() {
            match *token {
                Token::Plus => {
                    self.consume();
                    let right = self.parse_term()?;
                    left = Expr::Add(Box::new(left), Box::new(right));
                }
                Token::Minus => {
                    self.consume();
                    let right = self.parse_term()?;
                    left = Expr::Subtract(Box::new(left), Box::new(right));
                }
                _ => break,
            }
        }
        Ok(left)
    }

    fn parse_term(&mut self) -> Result<Expr, String> {
        let mut left = self.parse_factor()?;

        while let Some(token) = self.peek() {
            match *token {
                Token::Star => {
                    self.consume();
                    let right = self.parse_factor()?;
                    left = Expr::Multiply(Box::new(left), Box::new(right));
                }
                Token::Slash => {
                    self.consume();
                    let right = self.parse_factor()?;
                    left = Expr::Divide(Box::new(left), Box::new(right));
                }
                _ => break,
            }
        }
        Ok(left)
    }

    fn parse_factor(&mut self) -> Result<Expr, String> {
        let token = self.peek().cloned();
        match token {
            Some(Token::Number(n)) => {
                self.consume();
                Ok(Expr::Number(n))
            }
            Some(Token::Identifier(name)) => {
                self.consume();
                Ok(Expr::Variable(name))
            }
            Some(Token::LParen) => {
                self.consume();
                let expr = self.parse_expr()?;
                if self.peek_and_consume_if(|t| matches!(t, Token::RParen)).is_none() {
                    return Err("Expected closing parenthesis".to_string());
                }
                Ok(expr)
            }
            None => Err("Unexpected end of expression".to_string()),
            _ => Err("Unexpected token in factor".to_string()),
        }
    }
}

pub struct Evaluator;

impl Evaluator {
    pub fn evaluate(expr: &Expr, objects: &[HashMap<String, Value>]) -> Result<Value, String> {
        match expr {
            Expr::Aggregate(agg_type, inner) => {
                let values: Vec<f64> = objects.iter()
                    .filter_map(|obj| Self::evaluate_single(inner, obj).ok())
                    .filter_map(|v| match v {
                        Value::Float64(f) => Some(f),
                        Value::Int64(i) => Some(i as f64),
                        Value::UniqueId(u) => Some(u as f64),
                        _ => None
                    })
                    .collect();
                
                if values.is_empty() {
                    return Ok(Value::Null);
                }

                let result = match agg_type {
                    AggregateType::Sum => values.iter().sum(),
                    AggregateType::Mean => values.iter().sum::<f64>() / values.len() as f64,
                    AggregateType::Std => {
                        let mean = values.iter().sum::<f64>() / values.len() as f64;
                        let variance = values.iter()
                            .map(|x| (x - mean).powi(2))
                            .sum::<f64>() / values.len() as f64;
                        variance.sqrt()
                    },
                    AggregateType::Min => values.iter()
                        .fold(f64::INFINITY, |a, &b| a.min(b)),
                    AggregateType::Max => values.iter()
                        .fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
                };
                
                Ok(Value::Float64(result))
            }
            _ => {
                if objects.len() == 1 {
                    Self::evaluate_single(expr, &objects[0])
                } else {
                    Err("Expected single object for non-aggregate expression".to_string())
                }
            }
        }
    }

    fn evaluate_single(expr: &Expr, object: &HashMap<String, Value>) -> Result<Value, String> {
        match expr {
            Expr::Number(n) => Ok(Value::Float64(*n)),
            Expr::Variable(name) => object.get(name)
                .cloned()
                .ok_or_else(|| format!("Variable {} not found", name)),
            Expr::Add(left, right) => {
                match (Self::evaluate_single(left, object)?, Self::evaluate_single(right, object)?) {
                    (Value::Int64(a), Value::Int64(b)) => Ok(Value::Int64(a + b)),
                    (Value::Float64(a), Value::Float64(b)) => Ok(Value::Float64(a + b)),
                    (Value::Int64(a), Value::Float64(b)) => Ok(Value::Float64(a as f64 + b)),
                    (Value::Float64(a), Value::Int64(b)) => Ok(Value::Float64(a + b as f64)),
                    (Value::UniqueId(a), Value::UniqueId(b)) => Ok(Value::UniqueId(a + b)),
                    (Value::Null, _) | (_, Value::Null) => Ok(Value::Null),
                    (a, b) => Err(format!("Cannot add values: {:?} and {:?}", a, b))
                }
            },
            Expr::Subtract(left, right) => {
                match (Self::evaluate_single(left, object)?, Self::evaluate_single(right, object)?) {
                    (Value::Int64(a), Value::Int64(b)) => Ok(Value::Int64(a - b)),
                    (Value::Float64(a), Value::Float64(b)) => Ok(Value::Float64(a - b)),
                    (Value::Int64(a), Value::Float64(b)) => Ok(Value::Float64(a as f64 - b)),
                    (Value::Float64(a), Value::Int64(b)) => Ok(Value::Float64(a - b as f64)),
                    (Value::UniqueId(a), Value::UniqueId(b)) => Ok(Value::UniqueId(a - b)),
                    (Value::Null, _) | (_, Value::Null) => Ok(Value::Null),
                    (a, b) => Err(format!("Cannot subtract values: {:?} and {:?}", a, b))
                }
            },
            Expr::Multiply(left, right) => {
                match (Self::evaluate_single(left, object)?, Self::evaluate_single(right, object)?) {
                    (Value::Int64(a), Value::Int64(b)) => Ok(Value::Int64(a * b)),
                    (Value::Float64(a), Value::Float64(b)) => Ok(Value::Float64(a * b)),
                    (Value::Int64(a), Value::Float64(b)) => Ok(Value::Float64(a as f64 * b)),
                    (Value::Float64(a), Value::Int64(b)) => Ok(Value::Float64(a * b as f64)),
                    (Value::UniqueId(a), Value::UniqueId(b)) => Ok(Value::UniqueId(a * b)),
                    (Value::Null, _) | (_, Value::Null) => Ok(Value::Null),
                    (a, b) => Err(format!("Cannot multiply values: {:?} and {:?}", a, b))
                }
            },
            Expr::Divide(left, right) => {
                match (Self::evaluate_single(left, object)?, Self::evaluate_single(right, object)?) {
                    (_, Value::Int64(0)) | (_, Value::Float64(0.0)) | (_, Value::UniqueId(0)) => {
                        Ok(Value::Null)  // Return null for division by zero
                    },
                    (Value::Int64(a), Value::Int64(b)) => Ok(Value::Float64(a as f64 / b as f64)),
                    (Value::Float64(a), Value::Float64(b)) => Ok(Value::Float64(a / b)),
                    (Value::Int64(a), Value::Float64(b)) => Ok(Value::Float64(a as f64 / b)),
                    (Value::Float64(a), Value::Int64(b)) => Ok(Value::Float64(a / b as f64)),
                    (Value::UniqueId(a), Value::UniqueId(b)) => Ok(Value::Float64(a as f64 / b as f64)),
                    (Value::Null, _) | (_, Value::Null) => Ok(Value::Null),
                    (a, b) => Err(format!("Cannot divide values: {:?} and {:?}", a, b))
                }
            },
            Expr::Aggregate(_, _) => Err("Nested aggregates not supported".to_string()),
        }
    }
}