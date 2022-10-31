class Node {
	value: number;
	grad: number;
	op: string;
	_backward: ((this: Node, other?: Node) => void) | undefined;
	children: Array<Node>;
	visited: boolean;
	constructor(value = 0, grad = 0, first?: Node, second?: Node, op = "noop") {
		this.value = value;
		this.grad = grad;
		this.op = op;
		this.children = new Array<Node>();
		this.visited = false;
		if (first != undefined) {
			this.children.push(first);
		}
		if (second != undefined) {
			this.children.push(second);
		}
	}

	mul(this: Node, other: Node | number): Node {
		other = other instanceof Node ? other : new Node(other);
		let out: Node = new Node(
			this.value * other.value,
			0,
			this,
			other,
			"mul"
		);
		out._backward = () => {
			other = other as Node;
			this.grad += other.value * out.grad;
			other.grad += this.value * out.grad;
		};
		return out;
	}

	add(this: Node, other: Node | number): Node {
		other = other instanceof Node ? other : new Node(other);

		let out: Node = new Node(
			this.value + other.value,
			0,
			this,
			other,
			"add"
		);
		out._backward = () => {
			other = other as Node;
			this.grad += out.grad;
			other.grad += out.grad;
		};

		return out;
	}
	sub(this: Node, other: Node | number): Node {
		other = other instanceof Node ? other : new Node(other);

		return this.add(other.mul(-1));
	}
	pow(this: Node, other: number): Node {
		let out: Node = new Node(
			Math.pow(this.value, other),
			0,
			this,
			undefined,
			"pow"
		);
		out._backward = () => {
			this.grad += other * Math.pow(this.value, other - 1) * out.grad;
		};
		return out;
	}

	relu(this: Node): Node {
		let out: Node = new Node(this.value, 0, this, undefined, "relu");
		if (this.value < 0) out.value = 0;
		out._backward = () => {
			if (out.value > 0) {
				this.grad += out.grad;
			}
		};
		return out;
	}
	div(this: Node, other: Node | number): Node {
		other = other instanceof Node ? other : new Node(other);
		return this.mul(other.pow(-1));
	}
	backward(): void {
		let topo: Node[] = new Array();
		let stack = new Array();
		stack.push(this);
		const isLeafNode = (node: Node): Node | undefined => {
			if (node.children.length < 1) {
				return undefined;
			} else {
				for (let i = 0; i < node.children.length; i++) {
					if (!node.children[i].visited) {
						return node.children[i];
					}
				}
				return undefined;
			}
		};
		while (stack.length > 0) {
			const ret = isLeafNode(stack[stack.length - 1]);
			if (ret == undefined) {
				stack[stack.length - 1].visited = true;
				topo.push(stack.pop());
			} else {
				stack.push(ret);
			}
		}
		this.grad = 1;
		topo.reverse();
		topo.forEach((n) => {
			n.visited = false;
			if (n._backward) n._backward();
		});
	}
}

export default Node;
