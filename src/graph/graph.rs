use std::collections::HashMap;
use snafu::{Snafu};

use tch::Tensor;




pub struct Graph<T> { 
   adj_matrix: ndarray::Array2<T>,
   features: Option<Tensor>,
   node_mapping: Option<HashMap<usize, String>>
}

pub struct GraphBuilder<T> { 
   adj_matrix: ndarray::Array2<T>,
   features: Option<Tensor>,
   node_mapping: Option<HashMap<usize, String>>
}

impl<T> GraphBuilder<T> {
    pub fn new(adj_matrix: ndarray::Array2<T>) -> Self {
        GraphBuilder {
            adj_matrix,
            features: None, 
            node_mapping: None,
        }
    }
    pub fn features(mut self, features : Tensor) -> Self {
        self.features = Some(features);
        self
    }
    pub fn node_mapping(mut self, node_mapping: HashMap<usize, String>) -> Self {
        self.node_mapping = Some(node_mapping);
        self
    }
    pub fn build(self) -> GraphResult<T> {
        Ok(Graph {
            adj_matrix: self.adj_matrix,
            features: self.features,
            node_mapping: self.node_mapping,  
        })
    }
}


pub type GraphResult<T> = Result<Graph<T>, GraphError>;

#[derive(Debug, Snafu)]
pub enum GraphError {
    #[snafu(display("There was a problem with matching features to the underlying graph"))]
    FeatureError
}

#[cfg(test)]
mod graph_tests {
    use tch::kind;

    use super::*;
    #[test]
    fn graph_builder_test() {
        let adj_matrix: ndarray::Array2<f32> = ndarray::Array2::zeros((2,2));
        let node_mapping = HashMap::new();
        let features = tch::Tensor::zeros(&[2,2], kind::INT64_CPU);
        let _graph:GraphResult<f32> = GraphBuilder::new(adj_matrix).features(features).node_mapping(node_mapping).build();
    }
}