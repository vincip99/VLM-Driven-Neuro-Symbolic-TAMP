(define (problem manipulation-task)
  (:domain manipulation)
  (:objects
    red_can - obj
    blue_can - obj
    yellow_cube - obj
    sorting_bin - location
    pot - location
  )
  (:init
    (on-table red_can)
    (on-table blue_can)
    (on-table yellow_cube)
  )
  (:goal (and
    (on red_can sorting_bin)
    (on blue_can sorting_bin)
    (on yellow_cube pot)
  ))
)
