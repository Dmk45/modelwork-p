x = tensor([
    # ============================================
    # STOCK 1 (Apple) - 3 days × 3 features
    # ============================================
    [
        # Day 0
        [150.5,  152.3,  10000],
        #  ↑       ↑       ↑
        # open   close  volume
        
        # Day 1
        [152.3,  151.8,  12000],
        #  ↑       ↑       ↑
        # open   close  volume
        
        # Day 2
        [151.8,  155.2,  15000]
        #  ↑       ↑       ↑
        # open   close  volume
    ],
    
    # ============================================
    # STOCK 2 (Google) - 3 days × 3 features
    # ============================================
    [
        # Day 0
        [85.2,  86.1,  8000],
        #  ↑      ↑      ↑
        # open  close volume
        
        # Day 1
        [86.1,  85.5,  9000],
        #  ↑      ↑      ↑
        # open  close volume
        
        # Day 2
        [85.5,  87.3,  11000]
        #  ↑      ↑      ↑
        # open  close volume
    ]
])

# Shape: (batch=2, sequence=3, features=3)
print(f"Input shape: {x.shape}")
# Input shape: torch.Size([2, 3, 3])


h0 = torch.zeros(2, 2, 2)

h0 = tensor([
    # ============================================
    # LAYER 1 - Initial Hidden States
    # ============================================
    [
        # Stock 1 (Apple) - Layer 1
        [0.0000,  0.0000],
        #   ↑       ↑
        # hidden  hidden
        # feat 0  feat 1
        
        # Stock 2 (Google) - Layer 1
        [0.0000,  0.0000]
        #   ↑       ↑
        # hidden  hidden
        # feat 0  feat 1
    ],
    
    # ============================================
    # LAYER 2 - Initial Hidden States
    # ============================================
    [
        # Stock 1 (Apple) - Layer 2
        [0.0000,  0.0000],
        #   ↑       ↑
        # hidden  hidden
        # feat 0  feat 1
        
        # Stock 2 (Google) - Layer 2
        [0.0000,  0.0000]
        #   ↑       ↑
        # hidden  hidden
        # feat 0  feat 1
    ]
])

# Shape: (num_layers=2, batch=2, hidden=2)
print(f"h0 shape: {h0.shape}")
# h0 shape: torch.Size([2, 2, 2])
```

