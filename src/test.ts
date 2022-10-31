import Node from "./engine";

function test_basic() {
	let x = new Node(-4);
	let z = x.mul(2).add(2).add(x);
	let q = z.relu().add(z.mul(x));
	let h = z.mul(z).relu();
	let y = h.add(q.add(q.mul(x)));
	y.backward();

	if (x.grad == 46) return "pass";
	return "fail";
}

function test_more() {
	let ca = 138.83381924198252;
	let cb = 645.5772594752186;
	let a = new Node(-4.0);
	let b = new Node(2.0);
	let c = a.add(b);
	let d = a.mul(b).add(b.pow(3.0));
	c = c.add(c.add(1.0));
	c = c.add(1).add(c).add(a.mul(-1.0));
	d = d.add(d.mul(2.0)).add(b.add(a).relu());
	d = d.add(b.sub(a).relu()).add(d.mul(3.0));

	let e = c.sub(d);
	let f = e.pow(2);
	let g = f.div(2.0);
	g = g.add(new Node(10.0).div(f));
	g.backward();
	return Math.abs(a.grad - ca) < 0.000001 && Math.abs(b.grad - cb) < 0.000001
		? "pass"
		: "fail";
}

console.log(test_basic());
console.log(test_more());
