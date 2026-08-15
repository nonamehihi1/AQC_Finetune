import jax.numpy as jnp

def check_shapes(batch_size=None):
    if batch_size is None:
        observations = jnp.zeros((32,)) # state_dim=32
        bshape = ()
    else:
        observations = jnp.zeros((batch_size, 32))
        bshape = (batch_size,)
        
    num_samples = 8
    
    # Simulate network outputs
    # V network typically outputs (..., 1)
    v_k = jnp.zeros((*bshape, 1))
    
    # Q network outputs (*bshape, num_samples)
    q_k = jnp.zeros((*bshape, num_samples))
    
    # Current code does:
    v_k_expanded = jnp.expand_dims(v_k, axis=-1)
    
    try:
        a_k = q_k - v_k_expanded
        print(f"Batch={batch_size}: q_k {q_k.shape} - v_k_exp {v_k_expanded.shape} = a_k {a_k.shape}")
    except Exception as e:
        print(f"Batch={batch_size}: ERROR: {e}")

check_shapes(None)
check_shapes(4)
