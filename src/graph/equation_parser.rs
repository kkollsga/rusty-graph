// src/graph/equation_parser.rs

use std::collections::HashMap;

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
                '0'..='9' => {
                    let mut number = String::new();
                    while let Some(&c) = chars.peek() {
                        if c.is_digit(10) || c == '.' {
                            number.push(c);
                            chars.next();
                        } else {
                            break;
                        }
                    }
                    tokens.push(Token::Number(number.parse().unwrap()));
                }
                'a'..='z' | 'A'..='Z' => {
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

    fn parse_aggregate(&mut self) -> Result<Expr, String> {
        let token = self.peek().cloned();
        match token {
            Some(Token::Aggregate(agg_type)) => {
                self.consume();
                if let Some(Token::LParen) = self.peek() {
                    self.consume();
                    let expr = self.parse_expr()?;
                    match self.peek() {
                        Some(Token::RParen) => {
                            self.consume();
                            Ok(Expr::Aggregate(agg_type, Box::new(expr)))
                        }
                        _ => Err("Expected closing parenthesis after aggregate expression".to_string())
                    }
                } else {
                    Err("Expected opening parenthesis after aggregate function".to_string())
                }
            }
            _ => self.parse_expr()
        }
    }

    fn parse_expr(&mut self) -> Result<Expr, String> {
        let mut left = self.parse_term()?;

        while let Some(token) = self.peek().cloned() {
            match token {
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

        while let Some(token) = self.peek().cloned() {
            match token {
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
                match self.peek() {
                    Some(Token::RParen) => {
                        self.consume();
                        Ok(expr)
                    }
                    _ => Err("Expected closing parenthesis".to_string())
                }
            }
            _ => Err("Unexpected token in factor".to_string())
        }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn consume(&mut self) {
        if self.pos < self.tokens.len() {
            self.pos += 1;
        }
    }
}

pub struct Evaluator;

impl Evaluator {
    pub fn evaluate(expr: &Expr, objects: &[HashMap<String, f64>]) -> Result<f64, String> {
        match expr {
            Expr::Aggregate(agg_type, inner) => {
                let values: Vec<f64> = objects.iter()
                    .map(|obj| Self::evaluate_single(inner, obj))
                    .collect::<Result<Vec<f64>, String>>()?;
                
                if values.is_empty() {
                    return Err("No values to aggregate".to_string());
                }

                Ok(match agg_type {
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
                })
            }
            // Changed this section for non-aggregation expressions
            _ => {
                if objects.len() == 1 {
                    Self::evaluate_single(expr, &objects[0])
                } else {
                    // For multiple objects, evaluate the expression for each object
                    let values: Vec<f64> = objects.iter()
                        .map(|obj| Self::evaluate_single(expr, obj))
                        .collect::<Result<Vec<f64>, String>>()?;
                    
                    if values.is_empty() {
                        Err("No objects to evaluate".to_string())
                    } else {
                        // For non-aggregation, return the first result
                        Ok(values[0])
                    }
                }
            }
        }
    }

    fn evaluate_single(expr: &Expr, object: &HashMap<String, f64>) -> Result<f64, String> {
        match expr {
            Expr::Number(n) => Ok(*n),
            Expr::Variable(name) => object.get(name)
                .copied()
                .ok_or_else(|| format!("Variable {} not found", name)),
            Expr::Add(left, right) => Ok(
                Self::evaluate_single(left, object)? + 
                Self::evaluate_single(right, object)?
            ),
            Expr::Subtract(left, right) => Ok(
                Self::evaluate_single(left, object)? - 
                Self::evaluate_single(right, object)?
            ),
            Expr::Multiply(left, right) => Ok(
                Self::evaluate_single(left, object)? * 
                Self::evaluate_single(right, object)?
            ),
            Expr::Divide(left, right) => {
                let divisor = Self::evaluate_single(right, object)?;
                if divisor == 0.0 {
                    Err("Division by zero".to_string())
                } else {
                    Ok(Self::evaluate_single(left, object)? / divisor)
                }
            },
            Expr::Aggregate(_, _) => Err("Nested aggregates not supported".to_string()),
        }
    }
}