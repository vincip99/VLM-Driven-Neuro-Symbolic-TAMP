(define (problem manipulation-task)
  (:domain manipulation)
  (:objects
    red_can - obj
    blue_can - obj
    yellow_cube - obj
    purple_cube - obj
    pot - location
    sorting_bin - location
  )
  (:init
    (on-table red_can)
    (on-table blue_can)
    (on-table yellow_cube)
    (on-table purple_cube)
  )
  (:goal (and
    (on red_can pot)
    (on blue_can pot)
    (on yellow_cube sorting_bin)
    (on purple_cube sorting_bin)
  ))
)
