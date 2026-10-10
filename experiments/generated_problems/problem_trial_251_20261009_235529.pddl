(define (problem manipulation-task)
  (:domain manipulation)
  (:objects
    red_can - obj
    yellow_cube - obj
    blue_can - obj
    sorting_bin - location
    pot - location
    red_cube - obj
  )
  (:init
    (on-table red_cube)
    (on-table yellow_cube)
    (on-table red_can)
    (on-table blue_can)
  )
  (:goal (and
    (on red_can sorting_bin)
    (on yellow_cube pot)
    (on blue_can sorting_bin)
  ))
)
