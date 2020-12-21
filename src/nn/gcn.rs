use std::{convert::TryInto, ops::Mul};
use tch::{kind, nn::Init::KaimingUniform, nn::Module, Tensor};

use super::message_passing::MessagePassing;

struct GCN {
    in_features: usize,
    out_features: usize,
    weight: Tensor,
    bias: Tensor,
    linear: tch::nn::Linear,
}
impl GCN {
    fn new(in_features: usize, out_features: usize, store: &tch::nn::Path) -> Self {
        let mut weight = Tensor::zeros(
            &[
                in_features.try_into().unwrap(),
                out_features.try_into().unwrap(),
            ],
            (tch::Kind::Float, tch::Device::Cpu),
        );
        let mut bias = Tensor::zeros(
            &[1, out_features.try_into().unwrap()],
            (tch::Kind::Float, tch::Device::Cpu),
        );
        weight.init(tch::nn::Init::KaimingUniform);
        bias.init(tch::nn::Init::KaimingUniform);

        GCN {
            in_features,
            out_features,
            weight,
            bias,
            linear: tch::nn::linear(
                store,
                in_features as i64,
                out_features as i64,
                Default::default(),
            ),
        }
    }
}
impl MessagePassing for GCN {
    fn forward(&mut self, input: Tensor, edge_index: Tensor) -> Tensor {
        let x = self.linear.forward(&input);
        println!("{:?}", x.size());
        println!("{:?}", edge_index.size());
        let new_weight = self.propagate(edge_index, x);
        new_weight
    }
    fn get_features(&self) -> Tensor {
        todo!();
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::data::graph::GraphProp;
    #[test]
    pub fn cora_test() {
        let cora = crate::data::cora::Cora::new().unwrap();
        let vs = tch::nn::VarStore::new(tch::Device::Cpu);
        let mut gcn = GCN::new(cora.get_features().size()[1] as usize, 20, &vs.root());
        let out = gcn.forward(cora.get_features(), cora.get_adjacency());
        ()
    }
}
