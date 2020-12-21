use tch::Tensor;

pub trait GraphProp {
    fn get_adjacency(&self) -> Tensor;

    fn get_features(&self) -> Tensor;
}
