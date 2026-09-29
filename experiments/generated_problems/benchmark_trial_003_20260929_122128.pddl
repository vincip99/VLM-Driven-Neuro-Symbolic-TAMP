(define (problem trial-003)
  (:domain manipulation)
  (:objects
    blue_can - obj
    table - location
  )
  (:init
    (on-table blue_can)
  )
  (:goal (and
    (on blue_can table)
  ))
)
