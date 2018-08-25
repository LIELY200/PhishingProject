let node = function(nodeData) {
	this.right = undefined;
	this.left = undefined;
	this.data = nodeData;

	return {
		getRight: this.getRight.bind(this),
		getLeft: this.getLeft.bind(this),
		addRight: this.addRight.bind(this),
		addLeft: this.addLeft.bind(this),
		getData: this.getData.bind(this)
	}
}

node.prototype.getRight = function() {
	return this.right;
};

node.prototype.getLeft = function() {
	return this.left;
}

node.prototype.addLeft = function(data) {
	this.left = new node(data);
}

node.prototype.addRight = function(data) {
	this.right = new node(data);
}

node.prototype.getData = function() {
	return this.data;
}

let root = new node("root");

// level 1
root.addLeft("root->left");
root.addRight("root->right");

// right level 2
root.getRight().addRight("root->right->right");
root.getRight().addLeft("root->right->left");

// left level 2
root.getLeft().addRight("root->left->right");
root.getLeft().addLeft("root->left->left");

function printTree(root) {
	if (!root) {
		return;
	}

	printTree(root.getRight());
	console.log(root.getData());
	printTree(root.getLeft());
}

function bubbleSort(arr) {
	didSwap = true;
	n = arr.length;

	while(didSwap) {
		didSwap = false;
		n = n - 1;

		for (let i = 0; i < n; i++) {
			if (arr[i] < arr[i + 1]) {
				let temp = arr[i];
				arr[i] = arr[i + 1];
				arr[i + 1] = temp;

				didSwap = true;
			}
		}
	}

	return arr;
}


function selectionSort(arr) {
	let min_index = 0;
	for(let i = 0; i < arr.length; i++) {
		min_index = getMinIndex(arr, i);
		swap(arr, i, min_index);
	}

	return arr;
}

function getMinIndex(arr, start) {
	let min = start;
	for(let i = start + 1; i < arr.length; i++) {
		if (arr[min] > arr[i]) {
			min = i;
		}
	}
	return min;
}

function swap(arr, index1, index2) {
	let temp = arr[index1];
	arr[index1] = arr[index2];
	arr[index2] = temp;
}

