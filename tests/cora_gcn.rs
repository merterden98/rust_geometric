
use std::io::Error;

use rust_geometric::{self, nn::{gcn::GCN, message_passing::MessagePassing}, data::graph::GraphProp};
use tch::{IndexOp, Kind, Tensor, nn::{self, Module, OptimizerConfig}};

#[derive(Debug)]
struct GCNNet {
    conv : GCN,
    edge_index : tch::Tensor
}

impl Module for GCNNet {
    fn forward(&self, xs: &tch::Tensor) -> tch::Tensor {
        let x = self.conv.forward(xs, &self.edge_index);
        x
    }
}

impl GCNNet {
    pub fn new(in_features : i64, out_features : i64, edge_index : tch::Tensor, store: &tch::nn::Path) -> Self {
        GCNNet {
            conv : GCN::new(in_features, out_features, store),
            edge_index
        }        
    }
}

#[test]
fn cora_gcn() -> Result<(), Error>{
    let cora = rust_geometric::data::cora::Cora::new()?;
    let vs = tch::nn::VarStore::new(tch::Device::Cpu);
    let gcn = GCNNet::new(cora.get_features().size()[1], cora.num_labels(), cora.get_adjacency(),&vs.root());
    let mut optimizer = nn::Adam::default().build(&vs, 0.01).unwrap();
    let net = nn::seq().add(gcn).add_fn(|xs| xs.relu()); 
    for epoch in 1..10 {
        let mut sum_loss = 0.0;
        let mut loss = Tensor::zeros(&[1], tch::kind::DOUBLE_CPU);
        let forward = net.forward(&cora.get_features());
        for i in 0..forward.size()[0] {
            loss += forward.i(i).cross_entropy_for_logits(&cora.labels().onehot(7).i(i).to_kind(Kind::Int64));
        }

        sum_loss += f64::from(&loss);
        optimizer.backward_step(&loss);
        //let loss = forward.cross_entropy_for_logits(&cora.labels().onehot(7));
        println!("{:?}", sum_loss);
    }
    Ok(())
}