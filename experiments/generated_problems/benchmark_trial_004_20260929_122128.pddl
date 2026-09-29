(define (problem trial-004)
  (:domain manipulation)
  (:objects
    green_can - obj
    sorting_bin - location
  )
  (:init
    (on-table green_can)
  )
  (:goal (and
    (on green_can sorting_bin)
  ))
)
