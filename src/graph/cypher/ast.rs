// src/graph/cypher/ast.rs
// Full Cypher AST definitions

use crate::datatypes::values::Value;
use crate::graph::pattern_matching::Pattern;

// ============================================================================
// Top-Level Query
// ============================================================================

/// A complete Cypher query: a pipeline of clauses
#[derive(Debug, Clone)]
pub struct CypherQuery {
    pub clauses: Vec<Clause>,
}

/// Each clause in the query pipeline
#[derive(Debug, Clone)]
pub enum Clause {
    Match(MatchClause),
    OptionalMatch(MatchClause),
    Where(WhereClause),
    Return(ReturnClause),
    With(WithClause),
    OrderBy(OrderByClause),
    Skip(SkipClause),
    Limit(LimitClause),
    Unwind(UnwindClause),
    Union(UnionClause),
    #[allow(dead_code)]
    Create(CreateClause),
    #[allow(dead_code)]
    Set(SetClause),
    #[allow(dead_code)]
    Delete(DeleteClause),
}

// ============================================================================
// MATCH Clause
// ============================================================================

/// MATCH clause reuses the existing Pattern from pattern_matching.rs
#[derive(Debug, Clone)]
pub struct MatchClause {
    pub patterns: Vec<Pattern>,
}

// ============================================================================
// WHERE Clause
// ============================================================================

/// WHERE clause with a predicate expression tree
#[derive(Debug, Clone)]
pub struct WhereClause {
    pub predicate: Predicate,
}

/// Predicate expression tree supporting AND/OR/NOT and comparisons
#[derive(Debug, Clone)]
pub enum Predicate {
    Comparison {
        left: Expression,
        operator: ComparisonOp,
        right: Expression,
    },
    And(Box<Predicate>, Box<Predicate>),
    Or(Box<Predicate>, Box<Predicate>),
    Not(Box<Predicate>),
    IsNull(Expression),
    IsNotNull(Expression),
    In {
        expr: Expression,
        list: Vec<Expression>,
    },
    StartsWith {
        expr: Expression,
        pattern: Expression,
    },
    EndsWith {
        expr: Expression,
        pattern: Expression,
    },
    Contains {
        expr: Expression,
        pattern: Expression,
    },
}

/// Comparison operators
#[derive(Debug, Clone, PartialEq)]
pub enum ComparisonOp {
    Equals,        // =
    NotEquals,     // <>
    LessThan,      // <
    LessThanEq,    // <=
    GreaterThan,   // >
    GreaterThanEq, // >=
}

// ============================================================================
// Expressions
// ============================================================================

/// Expressions used in WHERE, RETURN, ORDER BY, WITH
#[derive(Debug, Clone)]
pub enum Expression {
    /// Property access: n.name, r.weight
    PropertyAccess {
        variable: String,
        property: String,
    },
    /// A variable reference: n, r
    Variable(String),
    /// Literal value
    Literal(Value),
    /// Function call: count(n), sum(n.age), collect(n.name)
    FunctionCall {
        name: String,
        args: Vec<Expression>,
        distinct: bool,
    },
    /// Arithmetic operations
    Add(Box<Expression>, Box<Expression>),
    Subtract(Box<Expression>, Box<Expression>),
    Multiply(Box<Expression>, Box<Expression>),
    Divide(Box<Expression>, Box<Expression>),
    /// Unary negation: -n.value
    Negate(Box<Expression>),
    /// Star (*) for count(*)
    Star,
    /// List literal [1, 2, 3]
    ListLiteral(Vec<Expression>),
    /// CASE expression
    /// Generic form: CASE WHEN pred THEN result ... ELSE default END
    /// Simple form:  CASE expr WHEN val THEN result ... ELSE default END
    Case {
        operand: Option<Box<Expression>>,
        when_clauses: Vec<(CaseCondition, Expression)>,
        else_expr: Option<Box<Expression>>,
    },
    /// Parameter reference: $param_name
    Parameter(String),
}

/// Condition in a CASE WHEN clause
#[derive(Debug, Clone)]
pub enum CaseCondition {
    /// Generic form: CASE WHEN predicate THEN ...
    Predicate(Predicate),
    /// Simple form: CASE expr WHEN value THEN ...
    Expression(Expression),
}

// ============================================================================
// RETURN Clause
// ============================================================================

/// RETURN clause: list of expressions with optional aliases
#[derive(Debug, Clone)]
pub struct ReturnClause {
    pub items: Vec<ReturnItem>,
    pub distinct: bool,
}

/// A single item in RETURN: expression AS alias
#[derive(Debug, Clone)]
pub struct ReturnItem {
    pub expression: Expression,
    pub alias: Option<String>,
}

// ============================================================================
// WITH Clause
// ============================================================================

/// WITH clause: same structure as RETURN, acts as intermediate projection
#[derive(Debug, Clone)]
pub struct WithClause {
    pub items: Vec<ReturnItem>,
    pub distinct: bool,
    pub where_clause: Option<WhereClause>,
}

// ============================================================================
// ORDER BY / SKIP / LIMIT
// ============================================================================

/// ORDER BY clause
#[derive(Debug, Clone)]
pub struct OrderByClause {
    pub items: Vec<OrderItem>,
}

/// Single ORDER BY item: expression + direction
#[derive(Debug, Clone)]
pub struct OrderItem {
    pub expression: Expression,
    pub ascending: bool,
}

/// SKIP clause
#[derive(Debug, Clone)]
pub struct SkipClause {
    pub count: Expression,
}

/// LIMIT clause
#[derive(Debug, Clone)]
pub struct LimitClause {
    pub count: Expression,
}

// ============================================================================
// UNWIND / UNION (Phase 3)
// ============================================================================

/// UNWIND clause: expand a list into rows
#[derive(Debug, Clone)]
pub struct UnwindClause {
    pub expression: Expression,
    pub alias: String,
}

/// UNION clause: combine result sets
#[derive(Debug, Clone)]
pub struct UnionClause {
    pub all: bool,
    pub query: Box<CypherQuery>,
}

// ============================================================================
// Mutation Clauses (Phase 3 - not yet implemented)
// ============================================================================

/// CREATE clause
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct CreateClause {
    pub patterns: Vec<Pattern>,
}

/// SET clause
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SetClause {
    pub items: Vec<SetItem>,
}

/// Single SET item
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum SetItem {
    Property {
        variable: String,
        property: String,
        expression: Expression,
    },
    Label {
        variable: String,
        label: String,
    },
}

/// DELETE clause
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct DeleteClause {
    pub detach: bool,
    pub expressions: Vec<Expression>,
}
