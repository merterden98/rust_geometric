use std::{collections::HashMap, convert::TryFrom, fs::File, io::BufRead, io::BufReader};

use crate::data;
use ndarray::{self, Array2};
use tch::Tensor;

pub struct Cora {
    features: Vec<(String, Tensor)>,
    adj_matrix: ndarray::Array2<f32>,
    name_to_enum: HashMap<u32, u32>,
    num_labels: i64,
    labels: Tensor
}

impl Cora {
    pub fn new() -> std::io::Result<Self> {
        let graph_file = File::open("/home/merden/Code/Rust/geometric/cora/cora.cites")?;
        let reader = BufReader::new(graph_file);
        let mut adj_list: Vec<(u32, u32)> = Vec::new();
        for line in reader.lines() {
            let line = line.unwrap();
            let edge: Vec<u32> = line
                .split_ascii_whitespace()
                .map(|x| x.parse::<u32>().unwrap())
                .collect();

            adj_list.push((edge[0], edge[1]));
        }
        let features_file = File::open("/home/merden/Code/Rust/geometric/cora/cora.content")?;
        let reader = BufReader::new(features_file);
        let mut feature_vec = Vec::new();
        let mut labels = Vec::new();
        let mut label_to_num : HashMap<String, i32> = HashMap::new();
        let mut i = 0;
        for line in reader.lines() {
            let line = line.unwrap();
            let features: Vec<String> = line
                .split_ascii_whitespace()
                .map(|x| x.to_string())
                .collect();
            let node = features[0].clone();
            let encoding: Vec<f32> = features[1..features.len() - 1]
                .to_vec()
                .iter()
                .map(|x| x.parse::<f32>().unwrap())
                .collect();
            let label = &features[features.len() -1];
            if !label_to_num.contains_key(label) {
                label_to_num.insert(label.clone(), i);
                i += 1;
            }
            labels.push((node.clone(), *label_to_num.get(label).unwrap()));
            feature_vec.push((node.clone(), Tensor::of_slice(encoding.as_slice())));
        }
        let mut name_to_enum: HashMap<u32, u32> = HashMap::new();
        let mut i = 0;
        for (e1, e2) in adj_list.iter() {
            if !name_to_enum.contains_key(&*e1) {
                name_to_enum.insert(*e1, i);
                i += 1;
            }
            if !name_to_enum.contains_key(&*e2) {
                name_to_enum.insert(*e2, i);
                i += 1;
            }
        }
        let adj_matrix = Cora::_get_matrix(&adj_list, &name_to_enum);
        let labels = Cora::_get_labels(&labels, &name_to_enum);
        Ok(Cora {
            features: feature_vec,
            adj_matrix,
            name_to_enum,
            num_labels: 7, //Hardcoded as we don't want to bother with computing this
            labels
        })
    }
    pub fn num_labels(&self) -> i64 {
        self.num_labels
    }
    pub fn len(&self) -> usize {
        self.name_to_enum.len()
    }
    pub fn labels(&self) -> &Tensor {
        &self.labels
    }
    fn _get_matrix(adj_list: &Vec<(u32, u32)>, name_to_enum: &HashMap<u32, u32>) -> Array2<f32> {
        let mut adj_matrix = Array2::<f32>::zeros((name_to_enum.len(), name_to_enum.len()));
        for (e1, e2) in adj_list.iter() {
            let i = name_to_enum.get(e1).unwrap();
            let j = name_to_enum.get(e2).unwrap();
            adj_matrix[[*i as usize, *j as usize]] = 1.;
        }
        adj_matrix
    }
    fn _get_labels(labels: &Vec<(String, i32)>, name_to_enum: &HashMap<u32, u32>) -> Tensor {
        let mut label_vec: Vec<(&u32, i32)> = labels.into_iter().map(|x| (name_to_enum.get(&x.0.parse::<u32>().unwrap()).unwrap(),x.1)).collect();
        label_vec.sort_by(|(n1, _), (n2, _)| n1.cmp(n2));
        let label_vec : Vec<i32> = label_vec.into_iter().map(|x| x.1).collect();
        Tensor::try_from(label_vec).unwrap()
    }
    pub fn _test_edge_exists(&self, e1: u32, e2: u32) -> bool {
        let i = *self.name_to_enum.get(&e1).unwrap() as usize;
        let j = *self.name_to_enum.get(&e2).unwrap() as usize;
        self.adj_matrix[[i, j]] == 1.
    }
}

impl data::graph::GraphProp for Cora {
    fn get_features(&self) -> Tensor {
        let mut ref_vec: Vec<(u32, &Tensor)> = Vec::new();
        for (name, t) in self.features.iter() {
            ref_vec.push((
                *self
                    .name_to_enum
                    .get(&name.parse::<u32>().unwrap())
                    .unwrap(),
                t,
            ));
        }
        ref_vec.sort_by(|(n1, _), (n2, _)| n1.cmp(n2));
        let features: Vec<Tensor> = ref_vec.iter().map(|(_, t)| t.view((1, 1433))).collect();
        let test = Tensor::cat(&features[..], 0);
        test
    }
    fn get_adjacency(&self) -> Tensor {
        let adj_matrix = Tensor::of_slice(self.adj_matrix.as_slice().unwrap());
        let dim = self.adj_matrix.dim().0 as i64;
        let adj_matrix = adj_matrix.view((dim, dim));
        adj_matrix.to_sparse()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::graph::GraphProp;
    #[test]
    fn test_new() -> std::io::Result<()> {
        let _cora: Cora = Cora::new()?;
        let _ = _cora.get_adjacency();
        let _ = _cora.get_features();
        Ok(())
    }
    #[test]
    fn test_print() {
        let _cora = Cora::new();
        match _cora {
            Ok(g) => assert!(g._test_edge_exists(35, 1033)),
            Err(_) => (),
        }
    }
    #[test]
    fn test_labels() -> std::io::Result<()>{
        let _cora = Cora::new()?;
        let labels = _cora.labels();
        labels.onehot(7).print();
        Ok(()) 
    }
}
