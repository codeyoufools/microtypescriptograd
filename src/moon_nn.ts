import Net from "./net";
import Node from "./engine";
import { data } from "./data";

function MaxMarginLoss(y: Node[], yhat: Node[]): Node {
	let out = new Node(1).sub(yhat[0].mul(y[0])).relu();
	return out;
}
function read_data() {
	let rows = data.split("\n");
	let res = new Array<Array<Node>>();
	rows.forEach((r: string) => {
		let str = r.split(",");
		let sample = new Array<Node>();
		sample.push(new Node(parseFloat(str[1])));
		sample.push(new Node(parseFloat(str[2])));
		res.push(sample);
	});
	return res;
}

function read_labels() {
	let rows = data.split("\n");
	let res: Node[][] = new Array();
	rows.forEach((r: string) => {
		let str = r.split(",");
		let label: Node[] = new Array();
		label.push(new Node(parseFloat(str[0]) * 2 - 1));
		res.push(label);
	});
	return res;
}

let train_data: Node[][] = read_data();
let train_labels: Node[][] = read_labels();

let nn = new Net();
nn.linear(2, 16);
nn.relu();
nn.linear(16, 16);
nn.relu();
nn.linear(16, 1);

function loss() {
	let losses: Node[] = new Array();
	let acc: number[] = new Array();
	for (let i = 0; i < 100; i++) {
		let yhat = nn.forward(train_data[i]);
		yhat[0].value > 0 == train_labels[i][0].value > 0
			? acc.push(1)
			: acc.push(0);
		losses.push(MaxMarginLoss(train_labels[i], yhat));
	}
	let dataloss = losses
		.reduce((p, c) => {
			return p.add(c);
		}, new Node(0))
		.mul(1 / losses.length);
	let paramLoss = nn
		.parameters()
		.reduce((p, c) => {
			return c.mul(c).add(p);
		}, new Node(0))
		.mul(1e-4);
	let totalLoss = dataloss.add(paramLoss);
	let totalAcc =
		acc.reduce((p, c) => {
			return p + c;
		}, 0) / acc.length;
	return { total: totalLoss, acc: totalAcc };
}

function optimize() {
	for (let i = 0; i < 100; i++) {
		let { total, acc } = loss();
		nn.zero_grad();
		total.backward();
		nn.update(1.0 - (0.9 * i) / 100);
		console.log("Loss for " + i + ": " + total.value);
		console.log("Accuracy for " + i + ": " + acc);
	}
}
optimize();
