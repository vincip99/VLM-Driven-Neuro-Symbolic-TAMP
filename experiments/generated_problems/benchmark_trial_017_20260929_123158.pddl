(define (problem manipulation-task)
  (:domain manipulation)
  (:objects
    green_can - obj
    red_can - obj
    purple_cube - obj
    sorting_bin - location
    pot - location
  )
  (:init
    (on-table green_can)
    (on-table red_can)
    (on-table purple_cube)
  )
  (:goal (and
    (on green_can sorting_bin)
    (on red_can pot)
    (on purple_cube sorting_bin)
  ))
)
