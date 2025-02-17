// src/datatypes/type_conversions.rs
use pyo3::prelude::*;
use pyo3::Bound;
use chrono::NaiveDate;
use pyo3::types::{PyDateTime, PyDateAccess};

pub fn to_u32(value: &Bound<'_, PyAny>) -> Option<u32> {
    if value.is_none() {
        return None;
    }
    if let Ok(val) = value.extract::<u32>() {
        return Some(val);
    }
    if let Ok(val) = value.extract::<i64>() {
        if val >= 0 && val <= u32::MAX as i64 {
            return Some(val as u32);
        }
    }
    if let Ok(val) = value.extract::<f64>() {
        if val.fract() == 0.0 && val >= 0.0 && val <= u32::MAX as f64 {
            return Some(val as u32);
        }
    }
    if let Ok(s) = value.extract::<String>() {
        if let Ok(val) = s.parse::<u32>() {
            return Some(val);
        }
        if let Ok(val) = s.parse::<f64>() {
            if val.fract() == 0.0 && val >= 0.0 && val <= u32::MAX as f64 {
                return Some(val as u32);
            }
        }
    }
    None
}

pub fn to_i64(value: &Bound<'_, PyAny>) -> Option<i64> {
    if value.is_none() {
        return None;
    }
    if let Ok(val) = value.extract::<i64>() {
        return Some(val);
    }
    if let Ok(val) = value.extract::<f64>() {
        if val.fract() == 0.0 && val >= i64::MIN as f64 && val <= i64::MAX as f64 {
            return Some(val as i64);
        }
    }    
    if let Ok(s) = value.extract::<String>() {
        if let Ok(val) = s.parse::<i64>() {
            return Some(val);
        }
        if let Ok(val) = s.parse::<f64>() {
            if val.fract() == 0.0 && val >= i64::MIN as f64 && val <= i64::MAX as f64 {
                return Some(val as i64);
            }
        }
    }
    None
}

pub fn to_f64(value: &Bound<'_, PyAny>) -> Option<f64> {
    if value.is_none() {
        return None;
    }
    if let Ok(val) = value.extract::<f64>() {
        if val.is_nan() {
            return None;
        }
        return Some(val);
    }
    if let Ok(val) = value.extract::<i64>() {
        return Some(val as f64);
    }
    if let Ok(s) = value.extract::<String>() {
        if let Ok(val) = s.parse::<f64>() {
            if val.is_nan() {
                return None;
            }
            return Some(val);
        }
    }
    None
}

pub fn to_datetime(value: &Bound<'_, PyAny>) -> Option<NaiveDate> {
    if value.is_none() {
        return None;
    }

    Python::with_gil(|_py| {
        if let Ok(ts) = value.downcast::<PyDateTime>() {
            NaiveDate::from_ymd_opt(ts.get_year(), ts.get_month() as u32, ts.get_day() as u32)
        } else {
            None
        }
    })
}

pub fn to_bool(value: &Bound<'_, PyAny>) -> Option<bool> {
    if value.is_none() {
        return None;
    }
    if let Ok(b) = value.extract::<bool>() {
        return Some(b);
    }
    if let Ok(s) = value.str() {
        match s.to_string().to_lowercase().as_str() {
            "true" | "1" | "yes" | "t" | "y" => return Some(true),
            "false" | "0" | "no" | "f" | "n" => return Some(false),
            _ => return None,
        }
    }
    None
}