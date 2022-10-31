import Node from "./engine";
class Neuron {
	input_size: number;
	weights: Node[];
	bias: Node;

	constructor(input_size: number) {
		this.input_size = input_size;
		this.weights = new Array<Node>();
		for (let i = 0; i < input_size; i++) {
			this.weights.push(new Node(Math.random() * 2 - 1));
		}
		this.bias = new Node(0);
	}
	forward(input: Node[]): Node {
		if (input.length != this.input_size) {
			console.log(
				"Input must be same size as the number of neuron weights"
			);
			console.log("input length: " + input.length);
			console.log("in size: " + this.input_size);
			throw Error();
		}
		return input
			.map((n, i) => {
				return n.mul(this.weights[i]);
			})
			.reduce((p, c) => {
				return p.add(c);
			}, new Node(0))
			.add(this.bias);
	}
	update(learning_rate: number): void {
		for (let i = 0; i < this.weights.length; i++) {
			this.weights[i].value -= this.weights[i].grad * learning_rate;
		}
		this.bias.value -= this.bias.grad * learning_rate;
	}
	zero_grad() {
		for (let i = 0; i < this.weights.length; i++) {
			this.weights[i].grad = 0;
		}
		this.bias.grad = 0;
	}
	parameters() {
		let params: Node[] = new Array();
		this.weights.map((w) => {
			params.push(w);
		});
		params.push(this.bias);
		return params;
	}
}
class Layer {
	constructor() {}
	forward(input: Node[]): any {}
	update(learning_rate: number): void {}
	zero_grad() {}
	parameters() {
		return Array<Node>();
	}
}
class DenseLayer extends Layer {
	neurons: Neuron[];
	in_size: number;
	size: number;

	constructor(in_size: number, size: number) {
		super();
		this.size = size;
		this.in_size = in_size;
		this.neurons = new Array<Neuron>();
		for (let i = 0; i < size; i++) {
			this.neurons.push(new Neuron(in_size));
		}
	}
	forward(input: Node[]): Node[] {
		let output = new Array<Node>();
		for (let i = 0; i < this.size; i++) {
			output.push(this.neurons[i].forward(input));
		}

		return output;
	}
	update(learning_rate: number): void {
		for (let i = 0; i < this.neurons.length; i++) {
			this.neurons[i].update(learning_rate);
		}
	}
	zero_grad() {
		for (let i = 0; i < this.neurons.length; i++) {
			this.neurons[i].zero_grad();
		}
	}
	parameters() {
		let params: Node[] = new Array();
		for (let i = 0; i < this.neurons.length; i++) {
			this.neurons[i].parameters().map((p) => params.push(p));
		}
		return params;
	}
}
class ReLULayer extends Layer {
	constructor() {
		super();
	}
	forward(input: Node[]): Node[] {
		let output = new Array<Node>();
		for (let i = 0; i < input.length; i++) {
			output.push(input[i].relu());
		}
		return output;
	}
}

class Net {
	layers: Layer[];
	constructor() {
		this.layers = new Array<Layer>();
	}
	linear(in_size: number, size: number): void {
		this.layers.push(new DenseLayer(in_size, size));
	}
	relu(): void {
		if (this.layers.length > 0) this.layers.push(new ReLULayer());
	}

	forward(input: Node[]): Node[] {
		for (let i = 0; i < this.layers.length; i++) {
			input = this.layers[i].forward(input);
		}
		return input;
	}
	update(learning_rate: number) {
		for (let i = 0; i < this.layers.length; i++) {
			this.layers[i].update(learning_rate);
		}
	}
	zero_grad() {
		for (let i = 0; i < this.layers.length; i++) {
			this.layers[i].zero_grad();
		}
	}
	parameters() {
		let params: Node[] = new Array();
		for (let i = 0; i < this.layers.length; i++) {
			this.layers[i].parameters().map((p) => {
				params.push(p);
			});
		}
		return params;
	}
}
export default Net;
