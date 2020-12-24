use std::{convert::TryInto};
use tch::{nn::Init::KaimingUniform, nn::Module, Tensor};

use super::message_passing::MessagePassing;

#[derive(Debug)]
pub struct GCN {
    in_features: i64,
    out_features: i64,
    weight: Tensor,
    bias: Tensor,
    linear: tch::nn::Linear,
}
impl GCN {
    pub fn new(in_features: i64, out_features: i64, store: &tch::nn::Path) -> Self {
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
        weight.init(KaimingUniform);
        bias.init(KaimingUniform);

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
    fn forward(&self, input: &Tensor, edge_index: &Tensor) -> Tensor {
        let x = self.linear.forward(&input);
        let new_weight = self.propagate(edge_index, &x);
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
    pub fn cora_forward_test() {
        let cora = crate::data::cora::Cora::new().unwrap();
        let vs = tch::nn::VarStore::new(tch::Device::Cpu);
        let gcn = GCN::new(cora.get_features().size()[1], 20, &vs.root());
        let _ = gcn.forward(&cora.get_features(), &cora.get_adjacency());
        ()
    }
}
