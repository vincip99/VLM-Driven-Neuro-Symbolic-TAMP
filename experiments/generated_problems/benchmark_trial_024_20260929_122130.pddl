(define (problem trial-024)
  (:domain manipulation)
  (:objects
    blue_can - obj
    yellow_cube - obj
    sorting_bin - location
    pot - location
  )
  (:init
    (on-table blue_can)
    (on-table yellow_cube)
  )
  (:goal (and
    (on blue_can sorting_bin)
    (on yellow_cube pot)
  ))
)
