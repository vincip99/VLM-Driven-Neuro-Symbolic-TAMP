(define (problem trial-026)
  (:domain manipulation)
  (:objects
    yellow_cube - obj
    blue_can - obj
    pot - location
    sorting_bin - location
  )
  (:init
    (on-table yellow_cube)
    (on-table blue_can)
  )
  (:goal (and
    (on yellow_cube pot)
    (on blue_can sorting_bin)
  ))
)
