from jax import numpy as jnp

def reflect_x(vec):
    return jnp.array([-vec[0], vec[1], vec[2]])

def reflect_y(vec):
    return jnp.array([vec[0], -vec[1], vec[2]])

def reflect_z(vec):
    return jnp.array([vec[0], vec[1], -vec[2]])

def rotate_around_x(vec):
    return jnp.array([vec[0], -vec[1], -vec[2]])

def rotate_around_y(vec):
    return jnp.array([-vec[0], vec[1], -vec[2]])

def rotate_around_z(vec):
    return jnp.array([-vec[0], -vec[1], vec[2]])

def invert(vec):
    return -vec

def identity(vec):
    return vec