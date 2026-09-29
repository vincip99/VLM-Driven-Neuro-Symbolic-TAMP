(define (problem trial-001)
  (:domain manipulation)
  (:objects
    red_can - obj
    sorting_bin - location
  )
  (:init
    (on-table red_can)
  )
  (:goal (and
    (on red_can sorting_bin)
  ))
)
