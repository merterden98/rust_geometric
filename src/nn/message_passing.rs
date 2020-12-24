use tch::Tensor;
pub trait MessagePassing{
    fn forward(&self, inputs: &Tensor, edge_index: &Tensor) -> Tensor;

    fn get_features(&self) -> Tensor;
    fn propagate(&self, edge_index: &Tensor, w: &Tensor) -> Tensor {
        let adj_matrix = edge_index;
        println!("{:?} {:?}", adj_matrix, w);
        adj_matrix.mm(&w)
    }
}

#[cfg(test)]
mod tests {
    use tch::{IndexOp, kind};

    use super::*;
    #[test]
    fn it_works() {
        let tensor = tch::Tensor::zeros(&[3, 3], kind::INT64_CPU);
        tensor.i(0).copy_(&tch::Tensor::of_slice(&[0, 1, 0]));
        tensor.i(1).copy_(&tch::Tensor::of_slice(&[1, 0, 0]));
        tensor.i(2).copy_(&tch::Tensor::of_slice(&[0, 1, 0]));
        let mut v: Tensor = tch::Tensor::zeros(&[3, 3], kind::INT64_CPU);
        &v.fill_diagonal_(1, true);
        let neighbors = tch::Tensor::transpose(&tensor, 0, 1);
        let neighbors = &neighbors.to_sparse().mm(&v);
        assert_eq!(&tensor.transpose(0, 1), neighbors);
    }
    struct TestProp();
    impl MessagePassing for TestProp {
        fn forward(&self, _inputs: &Tensor, _edge_index: &Tensor) -> Tensor {
            todo!();
        }
        fn get_features(&self) -> Tensor {
            let features = tch::Tensor::zeros(&[3, 1], kind::INT64_CPU);
            &features.i(0).copy_(&tch::Tensor::of_slice(&[8]));
            &features.i(1).copy_(&tch::Tensor::of_slice(&[2]));
            &features.i(2).copy_(&tch::Tensor::of_slice(&[3]));
            features
        }
    }
    #[test]
    fn propagate_test() {
        let tensor = tch::Tensor::zeros(&[3, 3], kind::INT64_CPU);
        tensor.i(0).copy_(&tch::Tensor::of_slice(&[0, 1, 0]));
        tensor.i(1).copy_(&tch::Tensor::of_slice(&[1, 0, 0]));
        tensor.i(2).copy_(&tch::Tensor::of_slice(&[0, 1, 0]));
    }
}
