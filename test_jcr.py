import jacks_car_rental as jcr

jack = jcr.Jcr()
s, t = 14, 14
jack.S[s, t] = 1.0
jack.P[s, t] = 2
jcr.to_draw_something(jack.S, 'init')
i_moved, j_moved, cost = jack.apply_policy(s, t)
jack2 = jcr.Jcr()
jack2.S[i_moved, j_moved] = 1
jcr.to_draw_something(jack2.S, 'after moving')
print(f"i, j, cost after applying policy: {i_moved, j_moved, cost}")
S_after_rent, reward = jack.rent_cars(i_moved, j_moved)
jcr.to_draw_something(S_after_rent, 'after renting')
S_after_return = jack.return_cars(S_after_rent)
jcr.to_draw_something(S_after_return, 'after returning')
