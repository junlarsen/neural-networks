# Neural Networks

This repository contains implementations of various neural network architectures written in plain Rust without any
external dependencies (apart from the standard library).

In order to properly understand how neural networks are built, I have decided to implement them from scratch without
any help from libraries.

## Code structure

The code explicitly defines all operations that are performed on the various structures in the `nn-core` crate. All
operations are explicitly defined as functions (no overloading of std::ops traits). This is to make it as clear as
possible to see what operations are performed on the various data structures.

Licensed under the MPL 2.0 license.
